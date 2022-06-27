import os
import cv2 as cv
import random
import time
seed = random.randint(1, 10000)
print('Random seed: {}'.format(seed))
random.seed(seed)
start=time.perf_counter()
#读取图片形成字典
path="scene_categories"
files=os.listdir(path)
print(files)
i=0
img_dict={}
for file in files:
    path_img=path+"/"+file
    imgs=os.listdir(path_img)
    for img in imgs:
        img_path=path_img+"/"+img
        img_dict[str(cv.imread(img_path))]=i
    i=i+1
print(len(img_dict))
# accuracy=0
# for i in range(100):#随机取出一百张图片，为其分配标签，判断准确率
#     index=random.randint(0,len(img_dict)-1)
#     label=random.randint(0,14)
#     if label==img_dict[list(img_dict.keys())[index]]:
#         accuracy=accuracy+1
# print("随机分类的准确率为",accuracy/100)
#生成测试集
accuracy=[]
test_img={}
num=0
for file in files:
    #从每个场景里取100张图片
    path_img=path+"/"+file
    imgs=os.listdir(path_img)
    acc=0
    for i in range(100):
        index=random.randint(0,len(imgs)-1)
        label=random.randint(0,14)
        img_path=path_img+"/"+imgs[index]
        test_img[str(cv.imread(img_path))]=label
        if label==num:
            acc=acc+1
    accuracy.append(acc/100)
    num=num+1
total_acc=0
for i in range(len(test_img)):
    if img_dict[list(test_img.keys())[i]]==test_img[list(test_img.keys())[i]]:
        total_acc=total_acc+1
print("每个场景的平均分类准确度为",accuracy)
print(len(test_img))
print("所有场景的平均分类度为",total_acc/len(test_img))
end=time.perf_counter()
print("运行时间{} s".format(end-start))