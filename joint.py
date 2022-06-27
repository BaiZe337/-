# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# if __name__ == '__main__':
#     top, bot, left, right = 0, 0, 230, 0
#     img1 = cv.imread('Panorama-two\p1.jpg')
#     img2 = cv.imread('Panorama-two\p2.jpg')
#     srcImg = cv.copyMakeBorder(img1, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
#     testImg = cv.copyMakeBorder(img2, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
#     img1gray = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)
#     img2gray = cv.cvtColor(testImg, cv.COLOR_BGR2GRAY)
#     sift = cv.SIFT().create()
#     # find the keypoints and descriptors with SIFT
#     kp1, des1 = sift.detectAndCompute(img1gray, None)
#     kp2, des2 = sift.detectAndCompute(img2gray, None)
#     # FLANN parameters
#     FLANN_INDEX_KDTREE = 1
#     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#     search_params = dict(checks=50)
#     flann = cv.FlannBasedMatcher(index_params, search_params)
#     matches = flann.knnMatch(des1, des2, k=2)
#
#     # Need to draw only good matches, so create a mask
#     matchesMask = [[0, 0] for i in range(len(matches))]
#
#     good = []
#     pts1 = []
#     pts2 = []
#     # ratio test as per Lowe's paper
#     for i, (m, n) in enumerate(matches):
#         if m.distance < 0.7 * n.distance:
#             good.append(m)
#             pts2.append(kp2[m.trainIdx].pt)
#             pts1.append(kp1[m.queryIdx].pt)
#             matchesMask[i] = [1, 0]
#
#     draw_params = dict(matchColor=(0, 255, 0),
#                        singlePointColor=(255, 0, 0),
#                        matchesMask=matchesMask,
#                        flags=0)
#     img3 = cv.drawMatchesKnn(img1gray, kp1, img2gray, kp2, matches, None, **draw_params)
#     # plt.imshow(img3, ), plt.show()
#
#     rows, cols = srcImg.shape[:2]
#     MIN_MATCH_COUNT = 10
#     if len(good) > MIN_MATCH_COUNT:
#         src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#         dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#         M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
#         warpImg = cv.warpPerspective(testImg, np.array(M), (testImg.shape[1], testImg.shape[0]),
#                                      flags=cv.WARP_INVERSE_MAP)
#
#         for col in range(0, cols):
#             if srcImg[:, col].any() and warpImg[:, col].any():
#                 left = col
#                 break
#         for col in range(cols - 1, 0, -1):
#             if srcImg[:, col].any() and warpImg[:, col].any():
#                 right = col
#                 break
#
#         res = np.zeros([rows, cols, 3], np.uint8)
#         for row in range(0, rows):
#             for col in range(0, cols):
#                 if not srcImg[row, col].any():
#                     res[row, col] = warpImg[row, col]
#                 elif not warpImg[row, col].any():
#                     res[row, col] = srcImg[row, col]
#                 else:
#                     srcImgLen = float(abs(col - left))
#                     testImgLen = float(abs(col - right))
#                     alpha = srcImgLen / (srcImgLen + testImgLen)
#                     res[row, col] = np.clip(srcImg[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)
#
#         # opencv is bgr, matplotlib is rgb
#         # res = cv.cvtColor(res, cv.COLOR_BGR2RGB)
#         # show the result
#         cv.imshow("img", res)
#         cv.waitKey(0)  # 等待按键按下
#
#     else:
#         print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))

# import os
# import sys
# import cv2
# #import win32ui
#
#
# # ? python基于Stitcher图像拼接
#
#
# def imgstitcher(imgs):  # 传入图像数据 列表[] 实现图像拼接
#     stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
#     _result, pano = stitcher.stitch(imgs)
#
#     if _result != cv2.Stitcher_OK:
#
#         print("不能拼接图片, error code = %d" % _result)
#         sys.exit(-1)
#
#     output = "G:\Xue-Mountain-Enterance"+"\\"+'result2' + '.png'
#     cv2.imwrite(output, pano)
#     print("拼接成功. %s 已保存!" % output)
#
#
# if __name__ == "__main__":
#     # imgPath为图片所在的文件夹相对路径
#     imgPath = 'G:\Xue-Mountain-Enterance'
#
#     imgList = os.listdir(imgPath)
#     imgs = []
#     for imgName in imgList:
#         pathImg = os.path.join(imgPath, imgName)
#         img = cv2.imread(pathImg)
#         if img is None:
#             print("图片不能读取：" + imgName)
#             sys.exit(-1)
#         imgs.append(img)
#
#     imgstitcher(imgs)  # 拼接
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# import numpy as np
# import cv2
#
#
# class Stitcher:
#
#     # 拼接函数
#     def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
#         # 获取输入图片
#         (imageB, imageA) = images
#         # 检测A、B图片的SIFT关键特征点，并计算特征描述子
#         (kpsA, featuresA) = self.detectAndDescribe(imageA)
#         (kpsB, featuresB) = self.detectAndDescribe(imageB)
#
#         # 匹配两张图片的所有特征点，返回匹配结果
#         M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
#
#         # 如果返回结果为空，没有匹配成功的特征点，退出算法
#         if M is None:
#             return None
#
#         # 否则，提取匹配结果
#         # H是3x3视角变换矩阵
#         (matches, H, status) = M
#         # 将图片A进行视角变换，result是变换后图片
#         result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
#         self.cv_show('result', result)
#         # 将图片B传入result图片最左端
#         result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
#         self.cv_show('result', result)
#         # 检测是否需要显示图片匹配
#         if showMatches:
#             # 生成匹配图片
#             vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
#             # 返回结果
#             return (result, vis)
#
#         # 返回匹配结果
#         return result
#
#     def cv_show(self, name, img):
#         cv2.imshow(name, img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#     def detectAndDescribe(self, image):
#         # 将彩色图片转换成灰度图
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#         # 建立SIFT生成器
#         descriptor = cv2.SIFT_create()
#         # 检测SIFT特征点，并计算描述子
#         (kps, features) = descriptor.detectAndCompute(image, None)
#
#         # 将结果转换成NumPy数组
#         kps = np.float32([kp.pt for kp in kps])
#
#         # 返回特征点集，及对应的描述特征
#         return (kps, features)
#
#     def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
#         # 建立暴力匹配器
#         matcher = cv2.BFMatcher()
#
#         # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
#         rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
#
#         matches = []
#         for m in rawMatches:
#             # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
#             if len(m) == 2 and m[0].distance < m[1].distance * ratio:
#                 # 存储两个点在featuresA, featuresB中的索引值
#                 matches.append((m[0].trainIdx, m[0].queryIdx))
#
#         # 当筛选后的匹配对大于4时，计算视角变换矩阵
#         if len(matches) > 4:
#             # 获取匹配对的点坐标
#             ptsA = np.float32([kpsA[i] for (_, i) in matches])
#             ptsB = np.float32([kpsB[i] for (i, _) in matches])
#
#             # 计算视角变换矩阵
#             (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
#
#             # 返回结果
#             return (matches, H, status)
#
#         # 如果匹配对小于4时，返回None
#         return None
#
#     def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
#         # 初始化可视化图片，将A、B图左右连接到一起
#         (hA, wA) = imageA.shape[:2]
#         (hB, wB) = imageB.shape[:2]
#         vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
#         vis[0:hA, 0:wA] = imageA
#         vis[0:hB, wA:] = imageB
#
#         # 联合遍历，画出匹配对
#         for ((trainIdx, queryIdx), s) in zip(matches, status):
#             # 当点对匹配成功时，画到可视化图上
#             if s == 1:
#                 # 画出匹配对
#                 ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
#                 ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
#                 cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
#
#         # 返回可视化结果
#         return vis
#
# import cv2
#
# # 读取拼接图片
# imageA = cv2.imread("G:\Panorama-two\street\p1.jpg")
# imageB = cv2.imread("G:\Panorama-two\street\p2.jpg")
#
# # 把图片拼接成全景图
# stitcher = Stitcher()
# (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
#
# # 显示所有图片
# cv2.imshow("Image A", imageA)
# cv2.imshow("Image B", imageB)
# cv2.imshow("Keypoint Matches", vis)
# cv2.imshow("Result", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
导入基本库
"""
import os
import cv2
import imutils
import numpy as np
import imutils

img_dir = "G:\Panorama-two\street"
names = os.listdir(img_dir)

images = []
for name in names:
    img_path = os.path.join(img_dir, name)  # 路径拼接
    image = cv2.imread(img_path)  # 读取图片
    images.append(image)
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
status, stitched = stitcher.stitch(images)
if status==0:
    print(1)
    cv2.imshow(stitched)
    cv2.imwrite('G:\Panorama-two\street\\result.jpg', stitched)


