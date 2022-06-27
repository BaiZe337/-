# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('SimpleAR\scene.jpg', 0)
# img2 = img.copy()
# template = cv2.imread('SimpleAR\\template.png', 0)
# w, h = template.shape[::-1]
# # All the 6 methods for comparison in a list
# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
# for meth in methods:
#     img = img2.copy()
#     # eval 语句用来计算存储在字符串中的有效 Python 表达式
#     method = eval(meth)
#     # 模板匹配
#     res = cv2.matchTemplate(img, template, method)
#     # 寻找最值
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#     print(min_val,ma)
#     # 使用不同的比较方法，对结果的解释不同
#
#     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#     cv2.rectangle(img, top_left, bottom_right, 255, 2)
#     plt.subplot(121), plt.imshow(res, cmap='gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122), plt.imshow(img, cmap='gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(meth)
#     plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
#
# #读入图像，截图部分作为模板图片
# img_src = cv2.imread('SimpleAR\scene.jpg')
# img_templ = cv2.imread('SimpleAR\\template.png')
# print('img_src.shape:',img_src.shape)
# print('img_templ.shape:',img_templ.shape)
#
# for method in range(6):
#     #模板匹配
#     result = cv2.matchTemplate(img_src, img_templ, method)
#     print('result.shape:',result.shape)
#     print('result.dtype:',result.dtype)
#     #计算匹配位置
#     min_max = cv2.minMaxLoc(result)
#     if method == 0 or method == 1:   #根据不同的模式最佳匹配位置取值方法不同
#         match_loc = min_max[2]
#     else:
#         match_loc = min_max[3]
#     #注意计算右下角坐标时x坐标要加模板图像shape[1]表示的宽度，y坐标加高度
#     right_bottom = (match_loc[0] + img_templ.shape[1], match_loc[1] + img_templ.shape[0])
#     print('result.min_max:',min_max)
#     print('match_loc:',match_loc)
#     print('right_bottom',right_bottom)
#     #标注位置
#     img_disp = img_src.copy()
#     cv2.rectangle(img_disp, match_loc,right_bottom, (0,255,0), 5, 8, 0 )
#     cv2.normalize( result, result, 0, 255, cv2.NORM_MINMAX, -1 )
#     cv2.circle(result, match_loc, 10, (255,0,0), 2 )
#     #显示图像
#     fig,ax = plt.subplots(2,2)
#     fig.suptitle('Method=%d'%method)
#     ax[0,0].set_title('img_src')
#     ax[0,0].imshow(cv2.cvtColor(img_src,cv2.COLOR_BGR2RGB))
#     ax[0,1].set_title('img_templ')
#     ax[0,1].imshow(cv2.cvtColor(img_templ,cv2.COLOR_BGR2RGB))
#     ax[1,0].set_title('result')
#     ax[1,0].imshow(result,'gray')
#     ax[1,1].set_title('img_disp')
#     ax[1,1].imshow(cv2.cvtColor(img_disp,cv2.COLOR_BGR2RGB))
#     #ax[0,0].axis('off');ax[0,1].axis('off');ax[1,0].axis('off');ax[1,1].axis('off')
#     plt.show()
import cv2 as cv
import numpy as np


# 鼠标操作，鼠标选中源图像中需要替换的位置信息
def mouse_action(event, x, y, flags, replace_coordinate_array):
    cv.resizeWindow("collect coordinate", 640, 480);
    cv.imshow('collect coordinate', img_dest_copy)
    if event == cv.EVENT_LBUTTONUP:
        # 画圆函数，参数分别表示原图、坐标、半径、颜色、线宽(若为-1表示填充)
        # 这个是为了圈出鼠标点击的点
        cv.circle(img_dest_copy, (x, y), 2, (0, 255, 255), -1)

        # 用鼠标单击事件来选择坐标
        # 将选中的四个点存放在集合中，在收集四个点时，四个点的点击顺序需要按照 img_src_coordinate 中的点的相对位置的前后顺序保持一致
        print(f'{x}, {y}')
        replace_coordinate_array.append([x, y])


if __name__ == '__main__':
    # 首先，加载待替换的源图像，并获得该图像的长度等信息,cv.IMREAD_COLOR 表示加载原图
    img_src = cv.imread('SimpleAR\monk.png', cv.IMREAD_COLOR)
    h, w, c = img_src.shape
    # 获得图像的四个边缘点的坐标
    img_src_coordinate = np.array([[x, y] for x in (0, w - 1) for y in (0, h - 1)])
    print(img_src_coordinate)
    # cv.imshow('replace', replace)

    print("===========================")

    # 加载目标图像
    img_dest = cv.imread('SimpleAR\scene.jpg', cv.IMREAD_COLOR)

    # 将源数据复制一份，避免后来对该数据的操作会对结果有影响
    img_dest_copy = np.tile(img_dest, 1)

    # 源图像中的数据
    # 定义一个数组，用来存放要源图像中要替换的坐标点，该坐标点由鼠标采集得到
    replace_coordinate = []
    cv.namedWindow('collect coordinate',0)
    cv.setMouseCallback('collect coordinate', mouse_action, replace_coordinate)
    while True:
        # 当采集到四个点后，可以按esc退出鼠标采集行为
        if cv.waitKey(20) == 27:
            break

    print(replace_coordinate)

    replace_coordinate = np.array(replace_coordinate)
    # 根据选中的四个点坐标和代替换的图像信息完成单应矩阵
    matrix, mask = cv.findHomography(img_src_coordinate, replace_coordinate, 0)
    print(f'matrix: {matrix}')
    perspective_img = cv.warpPerspective(img_src, matrix, (img_dest.shape[1], img_dest.shape[0]))
    print(perspective_img)
    cv.namedWindow("resized", 0);
    cv.resizeWindow("resized", 640, 480);
    cv.imshow("resized", perspective_img)
    cv.waitKey(0)

    # cv.imshow('threshold', threshold_img)
    # 降噪，去掉最大或最小的像素点
    retval, threshold_img = cv.threshold(perspective_img, 0, 255, cv.THRESH_BINARY)
    # 将降噪后的图像与之前的图像进行拼接
    cv.copyTo(src=threshold_img, mask=np.tile(threshold_img, 1), dst=img_dest)
    cv.copyTo(src=perspective_img, mask=np.tile(perspective_img, 1), dst=img_dest)
    cv.namedWindow("result", 0);
    cv.resizeWindow("result", 640, 480);
    cv.imshow('result', img_dest)
    cv.waitKey()
    cv.destroyAllWindows()
