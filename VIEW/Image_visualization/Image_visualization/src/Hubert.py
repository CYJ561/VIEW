import numpy as np
from K_means import kmeans_classify


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


# num : 元素个数
# x :簇的分类索引号
# 计算簇的关系矩阵
def distance_cluster(x, center,num):

    dist_k = distancematrix(center)  # k个簇中心距离矩阵
    C_matrix = np.zeros((num,2))
    M_matrix = np.zeros((num, num))
    # M_matrix[range(long), range(long)] = 0
    for j in range(num):
        C_matrix[j][0] = j
    for cluster in range(len(x)):  # 取一个簇
        for i in x[cluster]:
            C_matrix[i][1] = cluster
    for x in range(num):
        for y in range(num):
            if C_matrix[x][1] != C_matrix[y][1]:
                M_matrix[x][y] = dist_k[int(C_matrix[x][1])][int(C_matrix[y][1])]
    # print("M_matrix",M_matrix)
    return M_matrix


# original_coordinates 点的原坐标
# cluster_index 点的索引值
# center k个簇的中心点
# k 参数
def Hubert_cluster(original_coordinates, center,cluster_index, k):

    dist_ij = distancematrix(original_coordinates)  # d(i,j)
    num = original_coordinates.shape
    dist_mij = distance_cluster(cluster_index, center,num[0])  # d(m_i, m_j)
    M = k*(k-1)/2
    sum = 0
    sum1 = 0
    sum2 = 0
    for i in range(num[0]):
        for j in range(num[0]):
            sum += dist_ij[i][j] * dist_mij[i][j]
            sum1 += dist_ij[i][j] * dist_ij[i][j]
            sum2 += dist_mij[i][j] * dist_mij[i][j]
    r = 1/M * sum  # r
    M_p = 1/M * np.sum(dist_ij)  # Mp
    M_c = 1/M * np.sum(dist_mij)
    sigma_p = 1/M * sum1 - M_p * M_p
    sigma_p = abs(sigma_p)
    sigma_c = 1/M * sum2 - M_c * M_c
    sigma_c = abs(sigma_c)
    Co = abs(r - M_p * M_c) / ((sigma_p**0.5) * (sigma_c ** 0.5))

    return Co


if __name__ == '__main__':
    x = np.array([[1, 2], [1, 1], [6, 4], [2, 1], [6, 3], [5, 4]])
    a, b, cluster_index = kmeans_classify(x, 2)
    print(cluster_index)
    Co = Hubert_cluster(x, cluster_index, 2)
    print(Co)



