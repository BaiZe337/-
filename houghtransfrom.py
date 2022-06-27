#https://blog.csdn.net/Ray_Songaaa/article/details/107505451代码地址
#https://cn.bing.com/search?q=Hough+Transform%E6%A3%80%E6%B5%8B%E5%9B%BE%E5%83%8F%E4%B8%AD%E7%9A%84%E7%9B%B4%E7%BA%BF%E3%80%81%E5%9C%86python&qs=n&form=QBRE&sp=-1&pq=hough+transform%E6%A3%80%E6%B5%8B%E5%9B%BE%E5%83%8F%E4%B8%AD%E7%9A%84%E7%9B%B4%E7%BA%BF%E3%80%81%E5%9C%86py&sc=0-27&sk=&cvid=DAE890B1DD82454192F4C9AB12F0B05F搜索网址
#检测圆形
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Hough-data/circle/7.jpg")
# cv2.imshow("IMG", img)
# cv2.waitKey(0)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#灰度图像
result=cv2.GaussianBlur(gray,(5,5),0)
plt.subplot(121),plt.imshow(gray,'gray')
plt.xticks([]),plt.yticks([])
#hough transform
# cv2.imshow("IMG", result)
# cv2.waitKey(0)
result=cv2.Canny(result,10,220,5)
cv2.imshow("IMG", result)
cv2.waitKey(0)
circles1 = cv2.HoughCircles(result,cv2.HOUGH_GRADIENT,2,
                            15,param1=100,param2=150,minRadius=10,maxRadius=1000)
# 参数(输入图像，检测方法，dp为检测内侧圆心的累加器图像的分辨率于输入图像之比的倒数，
# minDist表示两个圆之间圆心的最小距离，param1表示传递给canny边缘检测算子的高阈值，param2越小，可以检测到
# 根本不存在的圆，越大检测的圆更接近完美的圆，minDadius圆半径的最小值，maxRadius圆半径的最大值)
circles = circles1[0,:,:]#提取为二维
circles = np.uint16(np.around(circles))#四舍五入，取整
for i in circles[:]:
    cv2.circle(img,(i[0],i[1]),i[2],(255,0,0),5)#画圆
    cv2.circle(img,(i[0],i[1]),2,(255,0,255),10)#画圆心

# plt.subplot(122),plt.imshow(img)
# plt.xticks([]),plt.yticks([])
# plt.savefig('G:\计算机视觉\代码\Hough-data\circle\\7circle.jpg')
# plt.show()

#检测直线
# import cv2
# import numpy as np
# img = cv2.imread("Hough-data\line\hourses.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# image=cv2.GaussianBlur(gray,(25,25),2)
# #esult=cv2.medianBlur(image,5)
# result = cv2.Canny(image,50, 250)
# cv2.namedWindow("IMG", 0);
# cv2.resizeWindow("IMG", 640, 480);
# cv2.imshow("IMG", result)
# cv2.waitKey(0)
# lines = cv2.HoughLinesP(result, 1, np.pi/180,1, minLineLength=1, maxLineGap=5)
# #image必须是二值图像，rho表示线段以像素为单位的距离精度，theta表示线段以弧度为单位的角度精度，thershod表示阈值
# #参数，超过设定阈值才被检测出线段，minLineLength表示线段以像素为单位的最小长度，同一方向上两条线段判定为一条直线
# #的最大允许间隔
# lines = lines[:, 0, :]
# for x1, y1, x2, y2 in lines:
#     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
# cv2.namedWindow("result", 0);
# cv2.resizeWindow("result", 640, 480);
# cv2.imshow('result', img)
#
# # cv2.imwrite("Hough-data\line\hourses.jpg",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
