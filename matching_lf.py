import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylsd
from pylsd.pylsd.pylsd.lsd import *
from PIL import Image
import tools
import time
import psutil

def imgBrightness(img1, c, b): 
    rows, cols, channels = img1.shape
    blank = np.zeros([rows, cols, channels], img1.dtype)
    rst = cv2.addWeighted(img1, c, blank, 1-c, b)
    return rst

def get_mask(img, kp):
    x,y,_ = img.shape
    mask = np.zeros((x,y))
    for k in kp:
        mask[k[0]-1][k[1]-1] = 1
    return mask

def line_det(img):
    kp=[]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),5)
    linesL = lsd(gray)
    for line in linesL:
        x1, y1, x2, y2 = map(int,line[:4])
        kp.append((y1, x1))
        kp.append((y2, x2))
#         kp.append((y1+(y2-y1)//2, x1+(x2-x1)//2))
    mask = get_mask(img, kp)
    line_mask = cv2.dilate(mask, np.ones((25, 25), np.uint8))
    return line_mask, kp
  
def cal_rep_(pt1, pt2, H, good_matches):
    count = 0
    for i in range (len(good_matches)):
        x1 = pt1[i]
        x2 = pt2[i]
        x_1to2 = np.dot(H, np.insert(np.array(x1).copy(),  2, 1, axis=0).T)
        d = np.sqrt((x_1to2[0]-x2[0])**2 + (x_1to2[1]-x2[1])**2)
        if (d <1.5):
            count = count + 1
#     print(count)
    rep = count / len(good_matches)
    return rep

path = './01.png'
img = cv2.imread(path)
dst1 = imgBrightness(img, 0.5, 0)  #dark
dst2 = imgBrightness(img, 1.5, 0)  #bright

line_mask, kp = line_det(img)
line_mask1, kp1 = line_det(dst1)
line_mask2, kp2 = line_det(dst2)

sifter = cv2.SIFT_create(100000000, contrastThreshold=-10000,edgeThreshold=-10000)
kps, desc = sifter.detectAndCompute(img, np.array(line_mask,dtype='uint8'))
kps1, desc1 = sifter.detectAndCompute(dst1, np.array(line_mask1,dtype='uint8'))
kps2, desc2 = sifter.detectAndCompute(dst2, np.array(line_mask2,dtype='uint8'))

matcher = cv2.BFMatcher()

raw_matches1 = matcher.knnMatch(desc, desc1, k = 2)
good_matches1 = []
pt1_1 = []
pt2_1 = []
for m1, m2 in raw_matches1:
    if m1.distance < 0.85 * m2.distance:
        pt1_1.append(kps[m1.queryIdx].pt)
        pt2_1.append(kps2[m1.trainIdx].pt)
        good_matches1.append([m1])

pt1= np.float32([kps[m[0].queryIdx].pt for m in good_matches1]).reshape(-1, 1, 2)
pt2 = np.float32([kps1[m[0].trainIdx].pt for m in good_matches1]).reshape(-1, 1, 2)
H1, mask = cv2.findHomography(ptsA_1, ptsB_2, cv2.RANSAC, 5.0)
cal_rep(pt1, pt2, H1, good_matches1)

matches1 = cv2.drawMatchesKnn(img.copy(), kps, dst1.copy(), kps1, good_matches1, None, flags = 2)

plt.figure()
plt.imshow(matches1)
