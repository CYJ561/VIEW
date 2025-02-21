import os
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
from unified_format import unified_format
from LabColorHistogram import LabColorHistogram
from isomap import isomap
from TSNE import tsne
import matplotlib.pyplot as plt
from K_means import kmeans_classify
from Hubert import Hubert_cluster
from overlap_Cv import overlap

global Lab
global res
global k


# 格式转换
def start():
    path = r"D:\\VIEW\\Image_visualization\\Image_visualization\\Source\\"  # 源
    path2 = r"D:\\VIEW\\Image_visualization\\Image_visualization\\Destination\\"  # 目的
    unifiedFormat = unified_format(path, path2)
    unifiedFormat.united()
    print("已统一为png格式")


# Lab特征提取图像集
def lab_feature():
    path = "D:\\VIEW\\Image_visualization\\Image_visualization\\Source\\"  # 目的
    LabHistogram = LabColorHistogram(path)
    global Lab  # 特征矩阵
    Lab = LabHistogram.Lab()
    print("特征矩阵", Lab)


# isosne算法：ISOMAP + T-SNE
def isosne_getdijk():

    # ISOMAP算法初步降维
    outcome = isomap(Lab, 2, 8)  # k = 8  # 改
    print("out的维度:", outcome.shape)
    print("out:", outcome)

    # t_sne = TSNE(n_components=2, random_state=0, perplexity=30.0, learning_rate=1500)
    # res = t_sne.fit_transform(img_feat)
    # 调用t-sne算法进行最终降维
    global res
    res = tsne(outcome, 2)  # 参数是待降维矩阵，维度2

    print("t-sne算法降维后数据维度:", res.shape)
    print("降维后数据坐标是:", res)
    # plt.scatter(res[:, 0], res[:, 1])
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('t-SNE')
    # plt.show()


# k-means聚类算法的调用和显示
def k_means():

    # 质心，簇分类，簇索引
    center, cluster, cluster_index = kmeans_classify(res, 8)  # 参数是点坐标阵，参数k
    print("中心", center)
    # print("簇分类", cluster)
    # print("簇索引", cluster_index)
    estimate_co(res, center,cluster_index, 8)  # Co
    estimate_cv(center, 8)  # Cv
    tx, ty = res[:, 0], res[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))
    bk_width = 4000
    bk_height = 3000
    max_dim = 200
    # 创建一幅给定模式（mode）和尺寸（size）的图片
    back = Image.new('RGB', (bk_width, bk_height), (255, 255, 255))
    # back1 = Image.new('RGB', (bk_width, bk_height), (0, 0, 0))

    path = "D:\\VIEW\\Image_visualization\\Image_visualization\\Source\\"  # 目的
    images_processed = os.listdir(path)  # 返回指定路径下的文件夹列表。

    for id, x, y in zip(images_processed, tx, ty):
        photo_file = path + id

        img = Image.open(photo_file)  # 图像读取
        rs = max(1, img.width / max_dim, img.height / max_dim)  # 比较选出最大值
        img = img.resize((int(img.width / rs), int(img.height / rs)))  # 该表图片大小
        back.paste(img, (int((bk_width - max_dim) * x), int((bk_height - max_dim) * y)))  # 在image的位置（x，y）处将b图像贴上去。

    back.show()

    # 聚类后的点阵显示
    data = res.copy()
    delt = []
    for index in center:  # del center
        # print(index)
        for c in range(len(data)):
            if (index == data[c]).all():
                delt.append(c)
    data = np.delete(data, delt, 0)
    # print(data)
    for i in range(len(data)):
        plt.scatter(data[i][0], data[i][1], marker='o', color='green', s=40, label='原始点')
        #  记号形状       颜色      点的大小      设置标签
    for j in range(len(center)):
        plt.scatter(center[j][0], center[j][1], marker='x', color='red', s=50, label='质心')
    plt.show()


# 评估Co
def estimate_co(res,center, cluster_index, k):

    # Co 是修改的Hubert统计量作为聚类度量
    Co = Hubert_cluster(res,center, cluster_index, k)  # 参数是点阵，簇索引号，参数k
    print('Co是：', Co)


# 评估Cv
def estimate_cv(c,k):

    Cv = overlap(c,k)
    print('Cv是：', Cv)


# 已经获得特征矩阵后的处理步骤
def find_k():

    kmax = 10  # 设置迭代最大数为10
    for k in range(2, 11):
        # 等度量特征映射算法的迭代
        outcome = isomap(Lab, 2, k)
        # t-sne算法迭代
        global res
        res = tsne(outcome, k)  # 点阵
        # 质心，簇分类，簇索引
        center, cluster, cluster_index = kmeans_classify(res, k)  # 参数是点坐标阵，参数k
        # Co 是修改的Hubert统计量作为聚类度量
        Co = Hubert_cluster(res, cluster_index, k)  # 参数是点阵，簇索引号，参数k
        Cv = overlap(res, k)
        C1 = []
        C1.append((Cv + Co) / 2)
        print(k, C1)
    plt.title('C1-k')
    plt.plot(range(2, kmax+1), C1, color='blue', label='C1-k')
    plt.legend()  # 显示图例
    plt.xlabel('k')
    plt.ylabel('C1')
    plt.savefig("C1-k.jpg")


if __name__ == '__main__':

    # start()  # 统一格式
    lab_feature()  # 提取特征向量
    isosne_getdijk()  # ISOSNE
    k_means()

