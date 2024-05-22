import os
import cv2
import glob
import numpy as np
import h5py
import math
import pydegensac as pyransac


def h5_to_dict(h5_file):
    ret = {}
    for k in h5_file:
        ret[k] = np.array(h5_file[k])
    return ret


def get_data_dir(data_dir='../data/phototourism/') -> str:
    return data_dir


def get_task_name(data_dir) -> list:
    task_dirs = glob.glob("{}/*".format(data_dir))
    task_names = [os.path.basename(i) for i in task_dirs]
    return task_names


def get_pairs_dict(data_dir) -> dict:
    task_names = get_task_name(data_dir)
    pairs_dict = {}
    for task_name in task_names:
        th_dir = os.path.join(data_dir, task_name, "set_100/new-vis-pairs/")
        ths = np.arange(10) / 10
        pair_by_th = {}
        for th in ths:
            file_name = os.path.join(th_dir, "keys-th-{}.npy".format(th))
            pair = np.load(file_name)
            pair_by_th['{:.1f}'.format(th)] = pair
        pairs_dict[task_name] = pair_by_th
    return pairs_dict


def get_imgs_dict(data_dir):
    task_names = get_task_name(data_dir)
    img_path_dict = {}
    for task_name in task_names:
        img_dir = os.path.join(data_dir, task_name, "set_100/images/")
        img_files = glob.glob(img_dir + '/*.png')
        img_path_dict[task_name] = {}
        for img_file in img_files:
            pair = img_file.split('/')[-1].split('.')[0]
            img_path_dict[task_name][pair] = img_file
    return img_path_dict



def get_calib_dict(data_dir):
    task_names = get_task_name(data_dir)
    calib_dict = {}
    for task_name in task_names:
        calib_dir = os.path.join(data_dir, task_name, "set_100/calibration/")
        calib_files = glob.glob(calib_dir + "/*.h5")
        calib_by_name = {}
        for calib_file in calib_files:
            img_name = os.path.basename(calib_file)
            img_name = img_name.replace('calibration_', '').replace('.h5', '')
            calib_by_name[img_name] = h5_to_dict(h5py.File(calib_file, 'r'))
        calib_dict[task_name] = calib_by_name
    return calib_dict


def get_patches_dict(patch_dir, data_dir):
    task_names = get_task_name(data_dir)
    patches_dict = {}
    for task_name in task_names:
        patches_h5 = h5py.File("{}/{}/patches.h5".format(patch_dir, task_name), "r")
        kps_h5 = h5py.File("{}/{}/keypoints.h5".format(patch_dir, task_name), "r")
        kp_scores_h5 = h5py.File("{}/{}/scores.h5".format(patch_dir, task_name), "r")
        patches_dict[task_name] = {}
        for name in patches_h5:
            patch = np.array(patches_h5[name])
            kp = np.array(kps_h5[name])
            kp_score = np.array(kp_scores_h5[name])
            patches_dict[task_name][name] = (patch, kp, kp_score)
    return patches_dict


def get_thresholds():
    return ["0.{}".format(i) for i in range(10)]


def normalize_keypoints(keypoints, K):
    '''Normalize keypoints using the calibration data.'''
    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])
    return keypoints


def get_F_matrix(kp1, kp2, match, ransac_method='pyransac'):
    match_kp1 = kp1[match[0]]
    match_kp2 = kp2[match[1]]
    cv2.setRNGSeed(42)
    F, mask_F = pyransac.findFundamentalMatrix(
        match_kp1,
        match_kp2,
        0.5,
        0.999999,
        100000,
        0,
        error_type='sampson',
        symmetric_error_check=True,
        enable_degeneracy_check=True
    )
    return F, mask_F


def estimate_E_matrix(kp1, kp2, match, F, mask_F, K1, K2):
    _E = np.matmul(np.matmul(K2.T, F), K1)
    _E = _E.astype('float64')
    indices = match[:, mask_F.flatten()]
    return _E, indices


def quaternion_from_matrix(matrix, isprecise=False):
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0

        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)

    return q


def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt) ** 2))
    err_q = np.arccos(1 - 2 * loss_q)

    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt) ** 2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    return err_q, err_t


def eval_essential_matrix(p1n, p2n, E, dR, dt):
    if len(p1n) != len(p2n):
        raise RuntimeError('Size mismatch in the keypoint lists')

    if p1n.shape[0] < 5:
        return np.pi, np.pi / 2

    if E.size > 0:
        _, R, t, _ = cv2.recoverPose(E, p1n, p2n)
        err_q, err_t = evaluate_R_t(dR, dt, R, t)
    else:
        err_q = np.pi
        err_t = np.pi / 2

    return err_q, err_t


def get_matches(desc1, desc2, both=True):
    if not both:
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(desc1, desc2, k=2)
        valid_matches = [i for i in matches if i[0].distance < 0.9 * i[1].distance]
        valid_matches = [[i[0].queryIdx, i[0].trainIdx] for i in valid_matches]
        return np.array(valid_matches, 'int32')
    else:
        matches1 = get_matches(desc1, desc2, False)
        matches2 = get_matches(desc2, desc1, False)
        outs = []
        for i, idx in enumerate(matches1):
            q_idx, t_idx = idx
            cur_idx = np.where(matches2[:, 0] == t_idx)[0]
            if len(cur_idx) > 0:
                if matches2[cur_idx[0]][1] == q_idx:
                    outs += [[q_idx, t_idx]]
        if len(outs) < 0:
            return matches1[:8]
        else:
            return np.array(outs)


def estimate_metric_from_desc(kp1, kp2, desc1, desc2, calib1, calib2):
    match = get_matches(desc1, desc2).T
    return estimate_metric(kp1, kp2, calib1, calib2, match)


def get_errR_errt(calib1, calib2, kp1, kp2, E, indices):
    kp1n = normalize_keypoints(kp1, calib1['K'])
    kp2n = normalize_keypoints(kp2, calib2['K'])
    R1, t1 = calib1['R'], calib1['T'].reshape((3, 1))
    R2, t2 = calib2['R'], calib2['T'].reshape((3, 1))
    dR = np.dot(R2, R1.T)
    dT = t2 - np.dot(dR, t1)
    err_q, err_t = eval_essential_matrix(kp1n[indices[0]], kp2n[indices[1]], E, dR, dT)
    return err_q, err_t


def estimate_metric(kp1, kp2, calib1, calib2, match):
    F, mask_F = get_F_matrix(kp1, kp2, match)
    E, inlier = estimate_E_matrix(kp1, kp2, match, F, mask_F, calib1['K'], calib2['K'])
    err_q, err_t = get_errR_errt(calib1, calib2, kp1, kp2, E, inlier)
    return E, inlier, err_q, err_t


def drawMatches(im1, im2, kp1, kp2, mask=None):
    kp1 = [cv2.KeyPoint(i[0], i[1], 0) for i in kp1]
    kp2 = [cv2.KeyPoint(i[0], i[1], 0) for i in kp2]
    if isinstance(mask, np.ndarray):
        kp1 = [i for idx, i in enumerate(kp1) if mask[idx] == 1]
        kp2 = [i for idx, i in enumerate(kp2) if mask[idx] == 1]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]
    im = cv2.drawMatches(im1, kp1, im2, kp2, matches, None)
    return im


import sys
import time


class ProcessBar():
    def reset(self, length):
        self._length = length
        self._start = time.time()

    def show(self, i, msg=""):
        percents = (i + 1) / self._length
        equal_length = int(50 * percents) * "="
        empty_length = (49 - int(50 * percents)) * " "
        elapsed_time = time.time() - self._start
        eta_time = elapsed_time / percents * (1 - percents)
        line_str = "[{}>{}] {}/{} {:.1f}% ETA:{:.2f}s {:.2f}s {}" \
            .format(equal_length, empty_length, i, self._length,
                    100 * percents, eta_time, elapsed_time, msg)
        sys.stdout.write("\r" + line_str)

    def summary(self, i, msg=""):
        line_str = "[{}] {} {} {:.2f}s {}".format(
            50 * "=", i, self._length, time.time() - self._start, msg)
        sys.stdout.write("\r{}\n".format(line_str))


pb = ProcessBar()
