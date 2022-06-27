import os
import time
from sklearn.neighbors import KNeighborsClassifier
#数据预处理，读取所需要的图片
def read_imgs(data_dir):
    imgs = os.listdir(data_dir)
    img_path=[]
    for img in imgs:
        if img!="Thumbs.db":
            img_path.append(data_dir+"/"+img)
    print(img_path)
    return img_path

data_dir = 'scene_categories/'
catalog = ['bedroom', 'CALsuburb', 'industrial', 'kitchen', 'livingroom',
           'MITcoast', 'MITforest', 'MIThighway', 'MITinsidecity', 'MITmountain',
           'MITopencountry', 'MITstreet', 'MITtallbuilding', 'PARoffice', 'store']

imgSet = [ read_imgs(data_dir + catalog[i]) for i in range(len(catalog))]
print ("Label\t\tcount")
print ("---------------------")
for i, item in enumerate(catalog):
    print ("%s\t\t%s" %(item, len(imgSet[i])))

#产生训练集和测试集
import random
"""
功能：产生训练集和测试集
输入：
    imgSet：包含所有物品种类的图片路径
    split：根据split进行划分训练集和测试集，
           表示训练集的比例
输出：
    train_datas：训练集数据，列表类型
    test_datas：测试集数据，列表类型
    train_labels：训练集标签，列表类型
    test_labels：测试集标签，列表类型
"""
def make_dataset(imgSet, split):
    train_datas=[]
    test_datas = []
    train_labels = []
    test_labels = []
    #用index来表示label，即三种场景标签如下：
    for index, item in enumerate(imgSet):
        random.shuffle(item) #将某种场景数据打乱
        interval = int(len(item) * split)
        train_item = item[:interval]
        test_item = item[interval:]
        train_datas += train_item
        test_datas += test_item
        train_labels += [index for _ in range(len(train_item))]
        test_labels += [index for _ in range(len(test_item))]
    return train_datas, test_datas, train_labels, test_labels

train_datas, test_datas ,train_labels, test_labels = make_dataset(imgSet, 0.5)
print(train_datas)
#特征提取
import cv2
#将RGB转换成灰度图像
def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
    return gray
"""
功能：提取一张灰度图的SURF特征
输入：
    gray_img：要提取特征的灰度图
输出：
    key_query：兴趣点
    desc_query：描述符，即我们最终需要的特征
"""
def gen_surf_features(gray_img):
    #400表示hessian阈值，一般使用300-500，表征了提取的特征的数量，
    #值越大得到的特征数量越少，但也越突出。
    surf = cv2.SIFT_create(400)
    key_query, desc_query = surf.detectAndCompute(gray_img, None)
    return key_query, desc_query

# #测试gen_surf_features的结果
# import matplotlib.pyplot as plt
# img = cv2.imread(train_datas[0])
# img = to_gray(img)
# key_query, desc_query = gen_surf_features(img)
# imgOut = cv2.drawKeypoints(img, key_query, None, (255, 0, 0), 4)
# plt.imshow(imgOut)
# plt.show()
#提取所有图像的特征
"""
功能：提取所有图像的SURF特征
输入：
    imgs：要提取特征的所有图像
输出：
    img_descs：提取的SURF特征
"""
def gen_all_surf_features(imgs):
    img_descs = []
    for item in imgs:
        img = cv2.imread(item)
        try:
            img = to_gray(img)
            key_query, desc_query = gen_surf_features(img)
            img_descs.append(desc_query)
        except:
            print("one_error")
    return img_descs

img_descs = gen_all_surf_features(train_datas)

#开始聚类
import numpy as np
from sklearn.cluster import MiniBatchKMeans

"""
功能：提取所有图像的SURF特征
输入：
    img_descs：提取的SURF特征
输出：
    img_bow_hist：条形图，即最终的特征
    cluster_model：训练好的聚类模型
"""
def cluster_features(img_descs, cluster_model):
    n_clusters = cluster_model.n_clusters #要聚类的种类数
    #将所有的特征排列成N*D的形式，其中N表示特征数，
    #D表示特征维度，这里特征维度D=64
    train_descs = [desc for desc_list in img_descs
                       for desc in desc_list]
    train_descs = np.array(train_descs)#转换为numpy的格式

    #判断D是否为128
    if train_descs.shape[1] != 128:
        raise ValueError('期望的SURF特征维度应为128, 实际为'
                         , train_descs.shape[1])
    #训练聚类模型，得到n_clusters个word的字典
    cluster_model.fit(train_descs)
    #raw_words是每张图片的SURF特征向量集合，
    #对每个特征向量得到字典距离最近的word
    img_clustered_words = [cluster_model.predict(raw_words)
                           for raw_words in img_descs]
    #对每张图得到word数目条形图(即字典中每个word的数量)
    #即得到我们最终需要的特征
    img_bow_hist = np.array(
        [np.bincount(clustered_words, minlength=n_clusters)
         for clustered_words in img_clustered_words])

    return img_bow_hist, cluster_model

K = 300 #要聚类的数量，即字典的大小(包含的单词数)
cluster_model=MiniBatchKMeans(n_clusters=K, init_size=3*K)
train_datas, cluster_model = cluster_features(img_descs,
                                              cluster_model)

"""
功能：将一张图片转化为直方图的形式
输入：
    img_path：一张图片
    cluster_model：已经训练好的聚类模型
输出：
    img_bow_hist：直方图向量
"""
def img_to_vect(img_path, cluster_model):
    """
    Given an image path and a trained clustering model (eg KMeans),
    generates a feature vector representing that image.
    Useful for processing new images for a classifier prediction.
    """

    img = cv2.imread(img_path)
    try:
        gray = to_gray(img)
        kp, desc = gen_surf_features(gray)
        clustered_desc = cluster_model.predict(desc)
        img_bow_hist = np.bincount(clustered_desc,
                                   minlength=cluster_model.n_clusters)
        # 转化为1*K的形式,K为字典的大小，即聚类的类别数
        return img_bow_hist.reshape(1, -1)
    except:
        print("one error")




# #SVM
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import LinearSVC
#
# """
# 功能：分类
# 输入：
#     train_datas：训练集，即最终的特征(所有图像的直方图集合)，
#                  要求是numpy.array类型
#     train_labels：训练集的label，要求是numpy.array类型
# 输出：
#     classifier：训练好的分类器
# """
# def run_svm(train_datas, train_labels):
#     classifier = OneVsRestClassifier(
#         LinearSVC(random_state=0)).fit(train_datas, train_labels)
#     return classifier
#
# #将训练集label转化为numpy.array类型
# train_labels = np.array(train_labels)
# classifier = run_svm(train_datas, train_labels)


"""
功能：对测试集数据进行预测，得到Accuracy
输入：
    test_datas：测试集数据，要求是numpy.array类型
    test_labels：测试集label，要求是numpy.array类型
输出：
    无返回值，输出Accuracy
"""
def test(test_datas, test_labels, cluster_model, classifier):
    print ("测试集的数量: ", len(test_datas))
    singel_pred=[]
    for i in range(15):
        singel_pred.append([])
    preds = []
    for item in test_datas:
        vect = img_to_vect(item, cluster_model)
        pred = classifier.predict(vect)
        preds.append(pred[0])
    for j in range(len(preds)):
        if preds[j]==test_labels[j]:
            singel_pred[test_labels[j]].append(1)
        else:
            singel_pred[test_labels[j]].append(0)
    singel_acc=[]
    for i in range(15):
        singel_acc.append(sum(singel_pred[i])/len(singel_pred[i]))
    print("每个场景的平均分类准确度为",singel_acc)

    preds = np.array(preds)
    idx = preds == test_labels
    accuracy = sum(idx)/len(idx)
    print ("所有场景的平均分类准确度是: ", accuracy)
from sklearn.cluster import KMeans
import pickle
class KMeanClassifier():
    """默认使用欧式距离"""
    def __init__(self, X_train: np.asarray, y_train: np.asarray,
                  savefile="./model.ckpt"):
        self.X_train = X_train
        self.y_train = y_train
        self.savefile = savefile
        if not os.path.exists(savefile):
            self.__calClassCenter()
        self.data = pickle.load(open(self.savefile,"rb"))

    # 2.训练样本按标签聚类，计算每个类的中心
    def __calClassCenter(self):
        # 按类别建立一个dict
        dataset={}
        for x,y in zip(self.X_train,self.y_train):
            if y not in dataset:
                dataset[y]=[]
            dataset[y].append(x)

        # 计算每个类别的中心
        data = {}
        center = []
        labels = []
        for label in dataset:
            # data[label]=np.mean(np.asarray(dataset[label]),0)
            labels.append(label)
            center.append(np.mean(np.asarray(dataset[label]),0))
            # center.append(np.median(np.asarray(dataset[label]),0))

        data["label"] = labels
        data["center"] = center

        # 将这个dict保存，下次就可以不用再重新建立(节省时间)
        pickle.dump(data,open(self.savefile,"wb"))
        # return data

    # 3.预测样本
    def predict(self,X_test: np.asarray)->np.asarray:
        labels = np.asarray(self.data["label"])
        center = np.asarray(self.data["center"])
        result_dist = np.zeros([len(X_test), len(center)])
        for i, data in enumerate(X_test):
            data = np.tile(data, (len(center), 1))
            distance = np.sqrt(np.sum((data - center) ** 2, -1))
            result_dist[i] = distance

        # 距离从小到大排序获取索引
        result_index = np.argsort(result_dist, -1)

        # 将索引替换成对应的标签，取距离最小对应的类别
        y_pred = labels[result_index][...,0]

        return y_pred

    # 4.计算精度信息
    def accuracy(self,y_true,y_pred)->float:
        return round(np.sum(y_pred == y_true) / len(y_pred),5)

start=time.perf_counter()
test_labels = np.array(test_labels)
# kmodel=KMeanClassifier(train_datas,train_labels)
kmodel=KNeighborsClassifier(n_neighbors=10)
kmodel.fit(train_datas,train_labels)
test(test_datas, test_labels, cluster_model, kmodel)
end=time.perf_counter()
print("运行时间{} s".format(end-start))




