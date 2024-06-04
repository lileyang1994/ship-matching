import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import tools
import cv2
import os
import numpy as np
import pickle
from tqdm import tqdm
import h5py
import torch
import torch.utils
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pydegensac as pyransac
import torch.nn as nn
from train import HardNet
import dill
import time
from pylsd.pylsd.pylsd.lsd import *


path1 = './1.png'
path2 = '=./1-1.png'

sifter = cv2.SIFT_create(100000000, contrastThreshold=-10000,edgeThreshold=-10000)

device = torch.device('CUDA')
model = HardNet() 
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load("./model/Hardnet_retrain_myloader/9.h5", map_location=device)['net'])  
model = model.module
model.eval()
model.cuda()
model = torch.nn.DataParallel(model)

def get_transforms():
    MEAN_IMAGE = 0.443728476019
    STD_IMAGE = 0.20197947209
    transform = transforms.Compose([
        transforms.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)),
        transforms.Lambda(lambda x: x.reshape(32, 32, 1)),    
        transforms.ToTensor(),          
        transforms.Normalize((MEAN_IMAGE, ), (STD_IMAGE, ))
    ])
    return transform

def compute_patch_mat(kp, size, angle, patch_size=32, factor=12):
    x, y = kp
    scale = factor * size / patch_size
    offset_mat = np.array([(1, 0, patch_size / 2 - x), (0, 1, patch_size / 2 - y)], 'float32')
    rotate_mat = cv2.getRotationMatrix2D((x, y), 0, 1 / scale)
    M = np.matmul(offset_mat, np.concatenate([rotate_mat, [[0, 0, 1]]]))
    return M

def extract_img(img, mask, patch_size=32, factor=10, keep_size=8000):
    kps = sifter.detect(img, mask)
    img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    scores = np.array([i.response for i in kps])
    sorted_arg = np.argsort(scores)[::-1][:keep_size]
    scores = scores[sorted_arg]
    sizes = np.array([i.size for i in kps])[sorted_arg]
    angles = np.array([i.angle for i in kps])[sorted_arg]
    kps = np.array([i.pt for i in kps])[sorted_arg]
    patches = []
    for kp, size, angle in zip(kps, sizes, angles):
        M = compute_patch_mat(kp, size, angle, patch_size, factor)
        patches.append(cv2.warpAffine(img, M, (patch_size, patch_size), 
                                      flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS,
                                      borderMode=cv2.BORDER_REPLICATE))
    return np.array(patches), kps, scores

def get_mask(img, kp):
    x,y,_ = img.shape
    mask = np.zeros((x,y))
    for k in kp:
        mask[min(k[0]-1,(x-1))][min(k[1]-1,(y-1))] = 1
    return mask

def line_det(path):
    kp=[]
    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),5)
    linesL = lsd(gray)
    for line in linesL:
        x1, y1, x2, y2 = map(int,line[:4])
        kp.append((y1, x1))
        kp.append((y2, x2))
#         kp.append((y1+(y2-y1)//2, x1+(x2-x1)//2))
    line_mask = get_mask(img, kp)
    line_mask = cv2.dilate(line_mask, np.ones((25, 25), np.uint8))
    return line_mask

def get_desc(patches):
    descs = []
    for i in range(0, len(patches), 256):
        data = patches[i:i + 256].cuda()
        features = model(data)
        descs += features.detach().cpu().numpy().tolist()
    desc = np.array(descs, 'float32')
    return descs

img1 = cv2.imread(path1)
img2 = cv2.imread(path2)
mask1 = np.ones((img1.shape[0], img1.shape[1]))
mask2 = np.ones((img2.shape[0], img2.shape[1]))

line_mask1 = line_det(path1)
line_mask2 = line_det(path2)

mask1[line_mask1==0] = 0
mask1 = mask1.astype(int)
mask1 = np.array(mask1, dtype='uint8')
mask2[line_mask2==0] = 0
mask2 = mask2.astype(int)
mask2 = np.array(mask2, dtype='uint8')

patches1, kps1, scores1 = extract_img(img1, mask1)
patches2, kps2, scores2 = extract_img(img2, mask2)
patches1 = torch.stack([get_transforms()(x) for x in patches1])
patches2 = torch.stack([get_transforms()(y) for y in patches2])

desc1 = get_desc(patches1)
desc2 = get_desc(patches2)
desc1 = np.array(desc1).astype(np.float32)
desc2 = np.array(desc2).astype(np.float32)

match_pair = tools.get_matches(desc1, desc2) 
match_kp1 = kps1[match_pair[:, 0]]
match_kp2 = kps2[match_pair[:, 1]]
H, mask = pyransac.findHomography(match_kp1, match_kp2)

plt.rcParams['font.size'] = 16
plt.rcParams['figure.figsize'] = (12, 12)
plt.subplot(221)
plt.imshow(mask1)
plt.subplot(222)
plt.imshow(mask2)

res = tools.drawMatches(img1, img2, match_kp1, match_kp2)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
