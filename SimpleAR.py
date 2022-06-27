import cv2
import numpy as np

img1 = cv2.imread("./SimpleAR/template.png", )
img2 = cv2.imread("./SimpleAR/scene.jpg")
img3 = cv2.imread("./SimpleAR/monk.png")

# 使用SIFT检测角点,创建角点检测器
sift = cv2.SIFT_create()
# 获取关键点和描述符
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

ratio = 0.7
FIANN_INDEX_KDTREE = 1
index_parames = dict(algorithm=FIANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_parames, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < ratio * n.distance:
        good.append(m)

cv2.namedWindow("img1", cv2.WINDOW_NORMAL)
cv2.resizeWindow('img1', 600, 600)

cv2.imshow("img1", img2)
cv2.waitKey(0)

pic_match = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
cv2.imshow("img1", pic_match)
cv2.waitKey(0)

src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 1.0)
# pic_match2 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, matchesMask=mask, flags=2)
# cv2.imshow("img1", pic_match2)
# cv2.waitKey(0)

H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
h, w, dim = img1.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, H)
img2_new = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
cv2.namedWindow("img1", 0);
cv2.resizeWindow("img1", 640, 480);
cv2.imshow("img1", img2_new)
cv2.waitKey(0)

h = img3.shape[0]
w = img3.shape[1]
src_pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
H1, mask1 = cv2.findHomography(src_pts, dst, cv2.RANSAC, 5.0)

perspective_img = cv2.warpPerspective(img3, H1, (img2.shape[1], img2.shape[0]))
cv2.imshow("img1", perspective_img)
cv2.waitKey(0)
print(perspective_img)

img3_gray = cv2.cvtColor(perspective_img, cv2.COLOR_BGR2GRAY)
ret, mask= cv2.threshold(img3_gray, 0, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
img2_bg = cv2.bitwise_and(img2, img2, mask=mask_inv)
img3_fg = cv2.bitwise_and(perspective_img, perspective_img, mask=mask)
dst = cv2.add(img2_bg, img3_fg)
cv2.imshow("img1", dst)
cv2.waitKey(0)
