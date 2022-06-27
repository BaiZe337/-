import math
import os
import pickle

import cv2
import numpy as np

def np_skew_symmetric(v):
    zero = np.zeros_like(v[:, 0])
    M = np.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)
    return M

def get_episym(x1, x2, dR, dt):
    num_pts = len(x1)
    # Make homogeneous coordinates
    x1 = np.concatenate([
        x1, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)
    x2 = np.concatenate([
        x2, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)

    # Compute Fundamental matrix
    dR = dR.reshape(1, 3, 3)
    dt = dt.reshape(1, 3)
    F = np.repeat(np.matmul(
        np.reshape(np_skew_symmetric(dt), (-1, 3, 3)),
        dR
    ).reshape(-1, 3, 3), num_pts, axis=0)

    x2Fx1 = np.matmul(x2.transpose(0, 2, 1), np.matmul(F, x1)).flatten()
    Fx1 = np.matmul(F, x1).reshape(-1, 3)
    Ftx2 = np.matmul(F.transpose(0, 2, 1), x2).reshape(-1, 3)

    ys = x2Fx1**2 * (
        1.0 / (Fx1[..., 0]**2 + Fx1[..., 1]**2) +
        1.0 / (Ftx2[..., 0]**2 + Ftx2[..., 1]**2))
    return ys.flatten()

def load_data(root, seq, mode, filename):
    in_file_name = os.path.join(root, seq, mode, filename)
    with open(in_file_name, "rb") as ifp:
        data = pickle.load(ifp)
        ifp.close()
    return data

def normalize_keypoints(keypoints, K):
    '''Normalize keypoints using the calibration data.'''
    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])
    return keypoints

def unnormalize_keypoints(keypoints, K):
    '''Undo the normalization of the keypoints using the calibration data.'''
    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = keypoints * np.array([[f_x, f_y]]) + np.array([[C_x, C_y]])
    return keypoints

def metric_prf(ys, y_hat, theta, gt_theta=1e-4):
    p = []
    r = []
    f = []
    num_sample = len(xs)
    for i in range(num_sample):
        label = (ys[i] < gt_theta).astype(int)
        mask = (y_hat[i] < theta).astype(int)
        precision = np.sum(label * mask, dtype=float) / np.sum(mask)
        recall = np.sum(label * mask, dtype=float) / np.sum(label)
        if math.isnan(precision):
            precision = 0
        if math.isnan(recall):
            recall = 0
        f1 = 2 * precision * recall / (precision + recall)
        if math.isnan(f1):
            f1 = 0
        p = np.append(p, precision)
        r = np.append(r, recall)
        f = np.append(f, f1)
    return p, r, f

def vis_pais(xs, ys, ind, mask):# ind is the index of xs
    # sift matches visualization
    i = 2*ind
    j = 2*ind + 1
    I1 = cv2.imread(os.path.join(root, img_path[i]))
    I2 = cv2.imread(os.path.join(root, img_path[j]))
    rows1 = I1.shape[0]
    cols1 = I1.shape[1]
    rows2 = I2.shape[0]
    cols2 = I2.shape[1]
    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
    # Place the first image to the left
    #  out[:rows1,:cols1] = np.dstack([img1, img1, img1])
    out[:rows1, :cols1, :] = I1
    # Place the next image to the right of it
    out[:rows2, cols1:, :] = I2
    select = int(xs[0][0].shape[0] / 10)
    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    corr1 = xs[ind][0, :, :2]
    corr2 = xs[ind][0, :, 2:]
    # corr1 = unnormalize_keypoints(corr1, K[i])
    # corr2 = unnormalize_keypoints(corr2, K[j])
    for i in range(select):
        # x - columns
        # y - rows
        (x1, y1) = corr1[i]
        (x2, y2) = corr2[i]
        if mask[i] == 0:
            cv2.circle(out, (int(np.round(x1)), int(np.round(y1))), 2, (0, 0, 255), 1)
            cv2.circle(out, (int(np.round(x2) + cols1), int(np.round(y2))), 2, (0, 0, 255), 1)
            cv2.line(out, (int(np.round(x1)), int(np.round(y1))), (int(np.round(x2) + cols1), int(np.round(y2))),
                     (0, 0, 255), 2, shift=0)
        else:
            cv2.circle(out, (int(np.round(x1)), int(np.round(y1))), 2, (0, 255, 0), 1)
            cv2.circle(out, (int(np.round(x2) + cols1), int(np.round(y2))), 2, (0, 255, 0), 1)
            cv2.line(out, (int(np.round(x1)), int(np.round(y1))), (int(np.round(x2) + cols1), int(np.round(y2))),
                     (0, 255, 0), 2, shift=0)
    out1 = cv2.resize(out, None, fx=0.7, fy=0.7)
    cv2.imshow('vis', out1)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()

root = 'EG-data'
mode = 'easy'
seq = 'colosseum_exterior'
KRT_img = load_data(root, seq, mode, 'KRT_img.pkl')
K = KRT_img['K']
R = KRT_img['R']
T = KRT_img['t']
img_path = KRT_img['img_path']
xs = load_data(root, seq, mode, 'xs.pkl')['xs'] # unnormalize [initail match] shape:pairnum*1*kp*4
ys = []# to save sym_epi_distance
for ii in range(0, len(K), 2): #步长为2 ii=0 2 4 6 8 10 12...
    jj = ii+1
    # normalize_keypoints
    match = xs[int(ii/2)][0]
    x1 = match[:, :2].reshape(-1, 2)# kp in first image
    x2 = match[:, 2:].reshape(-1, 2)# correspondent kp in second image
    kp1 = normalize_keypoints(x1, K[ii])
    kp2 = normalize_keypoints(x2, K[jj])
    # Get dR
    R_i = R[ii]
    R_j = R[jj]
    dR = np.dot(R_j, R_i.T)
    # Get dt
    t_i = T[ii].reshape([3, 1])
    t_j = T[jj].reshape([3, 1])
    dt = t_j - np.dot(dR, t_i)
    # compute F and sym_epi_distance
    geod_d = get_episym(kp1, kp2, dR, dt)
    ys += [geod_d]

y_hat = ys # use mismatch removal method to get y_hat
th = 1e-3 # threshold
p, r, f = metric_prf(ys, y_hat, th)
p_avg = np.mean(p)
r_avg = np.mean(r)
f_avg = np.mean(f)
print(p_avg, r_avg, f_avg)
# mismatch removal results
ind = 2 # a pair reslut, use loop to show more or save
mask = (y_hat[ind] < th).astype(int)
vis_pais(xs, ys, ind, mask)