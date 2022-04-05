from __future__ import division, print_function
from tools import pb
import sys
import time
from copy import deepcopy
import math
import argparse
import torch
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import cv2

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal(m.weight.data, gain=0.6)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return
class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x
        
class HardNet(nn.Module):
    """HardNet model definition
    """
    def __init__(self):
        super(HardNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)
        return    
    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(
            -1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)
        
if __name__ == "__main__":
    MODEL_NAME = "Hardnet_retrain_myloader"       #changed
    LOG_DIR = "./line_matching"
    MODEL_DIR = "/localfeature-models/{}".format(MODEL_NAME)
    # print(MODEL_DIR)
    import os
    if not os.path.exists("data/models/{}".format(MODEL_NAME)):
        os.makedirs("data/models/{}".format(MODEL_NAME))

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    writer = SummaryWriter("{}/{}".format(LOG_DIR, MODEL_NAME))

    norm_trans = transforms.Compose([
        # transforms.Lambda(lambda x: cv2.resize(x, (32, 32))),
        transforms.Lambda(lambda x: np.reshape(x, (32, 32, 3))),
        transforms.ToTensor(),
        transforms.Normalize((0.443728476019,), (0.20197947209,))])

    def create_indices(_labels):
        inds = dict()
        for idx, ind in enumerate(_labels):
            if ind not in inds:
                inds[ind] = []
            inds[ind].append(idx)
        return inds

    class MatchingDataset(torch.utils.data.Dataset):
        def __init__(self, folder, name='liberty', transform=norm_trans, training=True):
            self.set = torchvision.datasets.PhotoTour(
                root=folder, name=name, download=True)
            self.indices = create_indices(self.set.labels.numpy())
            self.classes = list(self.indices)
            self.training = training
            self.transform = transform

        def __getitem__(self, idx):
            if not self.training:
                id1, id2, label = self.set.matches[idx]
                im1, im2 = self.set[id1], self.set[id2]
                im1 = np.concatenate([im1, im1, im1]).reshape(3,64,64).transpose(1, 2, 0)
                im2 = np.concatenate([im2, im2, im2]).reshape(3,64,64).transpose(1, 2, 0)
                return self.transform(im1), self.transform(im2), label
            cls_idx = np.random.randint(len(self.indices))
            indices = self.indices[cls_idx]
            n1, n2 = np.random.choice(indices, size=2, replace=False)
            im1, im2 = self.set[n1], self.set[n2]
            im1 = np.concatenate([im1, im1, im1]).reshape(3,64,64).transpose(1, 2, 0)
            im2 = np.concatenate([im2, im2, im2]).reshape(3,64,64).transpose(1, 2, 0)
            return self.transform(im1), self.transform(im2)
        def __len__(self):
            if self.training:
                return 5000000
            else:
                return len(self.set.matches)

    def get_distance(f1, f2):
        repeat_f1 = f1.reshape(-1, 1, 128)
        repeat_f2 = f2.reshape(1, -1, 128)
        dist_mat = torch.sqrt(
            ((repeat_f1 - repeat_f2) ** 2).sum(2) + 1e-6) + 1e-8
        return dist_mat
    def hardnet_loss_func(out_anchor, out_positive):
        dist_mat_ap = get_distance(out_anchor, out_positive)
        pos_loss = torch.diag(dist_mat_ap)
        eye = torch.autograd.Variable(torch.eye(dist_mat_ap.size(1))).cuda()
        dist_without_diag = dist_mat_ap + eye * 10
        easy_neg_mask = dist_without_diag.lt(0.008).float() * 10
        dist_without_diag = dist_without_diag + easy_neg_mask
        neg_loss_a_to_p = torch.min(dist_without_diag, 1)[0]
        neg_loss_p_to_a = torch.min(dist_without_diag, 0)[0]
        neg_loss = torch.min(neg_loss_a_to_p, neg_loss_p_to_a)
        loss = torch.clamp(1.0 + pos_loss - neg_loss, min= 0.0)
        loss = torch.mean(loss)
        return loss

    def distance_matrix_vector(anchor, positive):
        d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
        d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1)
        eps = 1e-6
        return torch.sqrt((d1_sq.repeat(1, positive.size(0)) + torch.t(d2_sq.repeat(1, anchor.size(0)))
                        - 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0))+eps)
    def percentile(t, q):
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(int(k)).values.item()
        return result
    def sosnet_loss_func(out_anchor, out_positive):
        #SOS_reg
        eps = 10-8
        dist_matrix_a = distance_matrix_vector(out_anchor,out_anchor)+eps
        dist_matrix_b = distance_matrix_vector(out_positive,out_positive)+eps
        k_max = percentile(dist_matrix_b, 50)
        #print("k_max:", k_max)
        mask = dist_matrix_b.lt(k_max)
        dist_matrix_a = dist_matrix_a*mask.int().float()
        dist_matrix_b = dist_matrix_b*mask.int().float()
        SOS_temp = torch.sqrt(torch.sum(torch.pow(dist_matrix_a-dist_matrix_b, 2)))
        return torch.mean(SOS_temp)
    
    def get_lr(step):
        return 0.1 * (1 - step * 1024.0 / 50000000)

    def hard_net_baseline_func(out_a, out_p):
        return loss_HardNet(out_a, out_p,
                            margin=args.margin,
                            anchor_swap=args.anchorswap,
                            anchor_ave=args.anchorave,
                            batch_reduce= args.batch_reduce,
                            loss_type= args.loss)

    class Framework:
        def __init__(self, net):
            self.net = net
            self.optimizer = optim.SGD(net.parameters(), lr=0.1,
                                       momentum=0.9, dampening=0.9,
                                       weight_decay=0.0001)
            self.loss_fn = hardnet_loss_func    #changed
            self.step = 0

        def set_train(self):
            self.net.train()

        def set_eval(self):
            self.net.eval()

        def predict(self, output_anchor, output_positive):
            dist = torch.sqrt(
                torch.sum((output_anchor - output_positive) ** 2, 1))
            scores = (2 - dist) / 2
            return scores

        def optimize(self, data_anchor, data_positive):
            data_anchor, data_positive = data_anchor.to(
                'cuda'), data_positive.to('cuda')
            output_anchor = self.net(data_anchor)
            output_positive = self.net(data_positive)
            loss = self.loss_fn(output_anchor, output_positive)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.step += 1
            return loss.item()

        def validate(self, data_anchor1, data_anchor2, label):
            data_anchor1, data_anchor2 = data_anchor1.to(
                'cuda'), data_anchor2.to('cuda')
            label = label.cuda()
            output_anchor1 = self.net(data_anchor1)
            output_anchor2 = self.net(data_anchor2)
            scores = self.predict(output_anchor1, output_anchor2)
            acc = ((scores > 0.5).float() == label.float()).float().mean()
            return acc.item(), scores.cpu().detach().numpy()

        def set_lr(self, lr):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        def save(self, path):
            torch.save({"net": self.net.state_dict(), "step": self.step}, path)

        def load(self, path):
            state_dict = torch.load(path)
            self.net.load_state_dict(state_dict['net'])
            self.step = state_dict['step']

        def get_precisionAt95Recall(self, labels, scores):
            recall_point = 0.95
            pos_scores = scores[labels == 1]
            neg_scores = scores[labels == 0]
            th_index = int(recall_point * len(pos_scores))
            th = np.sort(pos_scores)[::-1][th_index]
            predicts = scores > th
            P = (predicts == 1).sum()
            TP = ((predicts == 1) * (labels == 1)).sum()
            FN = ((predicts == 0) * (labels == 1)).sum()
            precision = TP / P
            return precision

    train_set = MatchingDataset('../data/sets/')
    notredame_set = MatchingDataset(
        '../data/sets/', name='notredame', training=False)
    yosemite_set = MatchingDataset(
        '../data/sets/', name='yosemite', training=False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=1024, shuffle=True, num_workers=12)
    valid_loaders = [
        {
            "name": "notredame",
            "dataloader": torch.utils.data.DataLoader(notredame_set, batch_size=128, shuffle=False, num_workers=8)
        },
        {
            "name": "yosemite",
            "dataloader": torch.utils.data.DataLoader(yosemite_set, batch_size=128, shuffle=False, num_workers=8)
        }
    ]

    net = HardNet()    #changed
    net = torch.nn.DataParallel(net)
    net.cuda()
    framework = Framework(net)

    def train(epoch):

        framework.set_train()
        pb.reset(len(train_loader))
        train_iter = iter(train_loader)
        train_loss = 0.0
        train_total = 0
        for batch_idx in range(len(train_loader)):
            t1 = time.time()
            data_anchor, data_positive = next(train_iter)
            t1 = time.time() - t1
            lr = get_lr(framework.step)
            framework.set_lr(lr)
            loss = framework.optimize(data_anchor, data_positive)
            writer.add_scalar("train/loss", loss, framework.step)
            writer.add_scalar("train/lr", lr, framework.step)

            train_loss += loss
            train_total += 1
            pb.show(batch_idx, "loss:{:.3f}, time:{:.3f}".format(loss, t1))
        loss = train_loss / train_total
        writer.add_scalar("epoch/loss", loss, epoch)
        pb.summary(epoch, "EPOCH loss:{:.3f}".format(loss))
        print()

    def valid(epoch):
        framework.set_eval()
        for valid_loader in valid_loaders:
            name = valid_loader['name']
            valid_iter = iter(valid_loader['dataloader'])
            valid_acc = 0.0
            valid_total = 0
            labels, scores = [], []
            pb.reset(len(valid_loader['dataloader']))
            for batch_idx in range(len(valid_loader['dataloader'])):
                a1, a2, label = next(valid_iter)
                acc, step_scores = framework.validate(a1, a2, label)
                scores += step_scores.tolist()
                labels += label.numpy().tolist()

                valid_acc += acc
                valid_total += 1
                pb.show(batch_idx, "acc:{:.3f}".format(acc))
            labels = np.array(labels)
            scores = np.array(scores)
            valid_acc = valid_acc / valid_total
            valid_err = framework.get_precisionAt95Recall(labels, scores)
            writer.add_scalar("valid/{}/acc".format(name), valid_acc, epoch)
            writer.add_scalar(
                "valid/{}/precision@95".format(name), valid_err, epoch)
            pb.summary(
                epoch, "VALID {}, acc:{:.3f}, precision@95:{:.3f}".format(name, valid_acc, valid_err))
            print()

    RESTORE_EPOCH = -1
    EPOCH = 10
    if RESTORE_EPOCH >= 0:
        print("LOAD {}".format)
        framework.load(MODEL_DIR + "/{}.h5".format(RESTORE_EPOCH))
    for epoch in range(RESTORE_EPOCH + 1, EPOCH, 1):
        train(epoch)
        valid(epoch)
        framework.save(MODEL_DIR + "/{}.h5".format(epoch))
        writer.flush()
