import math
import os
import numpy as np
from PIL import Image


# 计算距离矩阵
def distancematrix(test):
    length = len(test)   # 获得矩阵行数
    resmat = np.zeros([length, length], np.float32)  # 构建length*length的方阵（0填充）,类型是float32
    # 返回给定形状和类型的新数组，用0填充
    for i in range(length):
        for j in range(length):
            # np.linalg.norm()用于求范数
            # np.linalg.norm(x, ord=None, axis=None, keepdims=False)
            # x表示矩阵，ord表示范数类型
            # ord=1：表示求列和的最大值
            # ord=2：|λE-ATA|=0，求特征值，然后求最大特征值得算术平方根
            # ord=∞：表示求行和的最大值
            # ord=None：表示求整体的矩阵元素平方和，再开根号
            resmat[i][j] = np.linalg.norm(test[i] - test[j])  # 记录i行和j行之间矩阵的距离，整体的矩阵元素平方和，再开根号
    # print("resmat,距离矩阵是:", resmat)
    return resmat  # 返回距离矩阵


# 设置图片大小
def set_width(original_coordinates):

    tx, ty = original_coordinates[:, 0], original_coordinates[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))
    bk_width = 4000
    bk_height = 3000
    max_dim = 200
    w = 500
    # 创建一幅给定模式（mode）和尺寸（size）的图片
    back = Image.new('RGB', (bk_width, bk_height), (0, 0, 0))

    # 更改图片大小（w，w)
    path = "F:\\PythonPro\\Image_visualization\\Destination\\"  # 目的
    images_processed = os.listdir(path)  # 返回指定路径下的文件夹列表。

    for id, x, y in zip(images_processed, tx, ty):

        photo_file = path + id  # 图像绝对路劲
        img = Image.open(photo_file)  # 图像读取
        img = img.resize((w, w))  # 该表图片大小
        back.paste(img, (int((bk_width - max_dim) * x), int((bk_height - max_dim) * y)))  # 在image的位置（x，y）处将b图像贴上去。


# original_coordinates：点阵坐标
# 重叠函数的设计
def overlap(original_coordinates, k):

    R = 500
    # 距离矩阵dist
    dist_ij = distancematrix(original_coordinates)
    length = len(dist_ij)   # 获得矩阵行数
    for i in range(length):
        for j in range(length):
            if dist_ij[i][j] >= 2*R:  # 设置为0
                dist_ij[i][j] = 0
            else:
                d = dist_ij[i][j] / (2 * R)
                d = 2 * math.acos(d) - math.sin(2 * math.acos(d))
                d = R * R * d
                dist_ij[i][j] = d
    dist_ij[range(length), range(length)] = 0
    Cv = 1 / (k * (k-1)) * np.sum(1 - dist_ij / (np.pi * R * R))
    return Cv







