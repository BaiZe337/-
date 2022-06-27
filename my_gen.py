import os
import pickle
import cv2
import numpy as np


def load_data(root, seq, mode):
    in_file_name = os.path.join(root, seq, mode, 'KRT_img.pkl')
    with open(in_file_name, "rb") as ifp:
        data = pickle.load(ifp)
        ifp.close()
    return data


root = 'EG-data'
mode = 'easy'
seq = 'colosseum_exterior'
KRT_img = load_data(root, seq, mode)
K = KRT_img['K']
R = KRT_img['R']
T = KRT_img['t']
img_path = KRT_img['img_path']  # seq+mode+img_name

kp_num = 2000
sift = cv2.SIFT_create(nfeatures=kp_num, contrastThreshold=1e-5)


#暴力匹配法
flann = cv2.BFMatcher(cv2.NORM_L2)

# 图像匹配
kp = []
desc = []
xs = []

for ii in range(0, 20, 2):
    jj = ii + 1
    print("extract keypoint of image {}".format(ii))
    img_path1 = os.path.join(root, img_path[ii])
    img = cv2.imread(img_path1)
    kp1, des1 = sift.detectAndCompute(img, None)
    print("extract keypoint of image {}".format(jj))
    img_path2 = os.path.join(root, img_path[jj])
    img = cv2.imread(img_path2)
    kp2, des2 = sift.detectAndCompute(img, None)
    matches = flann.match(des1, des2)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 1.0)
    x1 = np.array([kp1[m.queryIdx].pt for m in matches])
    x2 = np.array([kp2[m.trainIdx].pt for m in matches])
    xs += [np.concatenate([x1, x2], axis=1).T.reshape(4, 1, -1).transpose(1, 2, 0)]
mydict = {'xs': xs}
out_file_name = os.path.join(root, seq, mode, 'xs.pkl')
with open(out_file_name, "wb") as ofp:
    pickle.dump(mydict, ofp)
    ofp.close()
print("saved kp info in {}".format(out_file_name))




