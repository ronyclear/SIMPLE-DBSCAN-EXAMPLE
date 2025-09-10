from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


# dbscan思路为：
# 1.准备待聚类数据data，并初始化visit_list为全0，shape与data的数据个数一直，初始化class_list为全-1(-1表示噪声，0,1,2表示类别),label=-1
# 2.遍历待聚类数据
# 3.若当前数据已被访问，则跳过, if visit_list[i]==1
# 4.若当前数据未被访问，则找出邻域eps内的点，若点数大于minPts，则该点为核心点，更新label=label+1, class_list[i] = label
# 5.将邻域内的未被访问的可达点作为search_list（除自身外）
# 6.若search_list不为空：while(len(search_list) > 0)
# 7.从search_list中选择一个点，并从list中将其剔除
# 8.判断该点是噪点，获取该点下标更在class_list中更新其label
# 9.若该点未被访问，则判断该点领域内的点的数量是否小于min_pts；
# 10.将其中已被访问并且是噪声点筛选出来
# 11.将这些噪声点对应的标签换为label
# 12.将领域内未被访问的点提取出来，将其添加至search_list后面并去重，重新赋值search_list
# -------------------
# 总体思想为：
# 找到一个领域内点数大于minpts的点，修改内点的类别，然后遍历领域内点列表search_list，遍历的点冲search中去除，
# 判断内点是否是核心，是的话则将内心的联通点叠加到search_list后面并去重

X1, y1=datasets.make_circles(n_samples=5000, factor=.5,noise=.05)

# 添加了噪声
X2, y2 = datasets.make_blobs(n_samples=500
                             ,n_features=2
                             ,centers=[[1.5,1.5]]
                             ,cluster_std=[[.1]]
                             ,random_state=0)

X = np.concatenate((X1, X2))  # 数据+噪声

n_sample = X.shape[0]
class_list = np.zeros((n_sample, 1)) # 初始化为全0，-1为噪声，0,1,2,3为类别
visit_list = np.zeros((n_sample, 1)) # 0表示未被访问，1表示被访问

# 标签初始化为-1
label = -1
eps = 0.15
min_pts = 2

for i in range(n_sample):
    # 若当前点已经被访问，则调过此次循环
    if visit_list[i] == 1:
        continue
    label += 1
    curr_pt = X[i]
    x_delta = X[:, 0] - curr_pt[0]
    y_delta = X[:, 1] - curr_pt[1]
    # 计算当前点和其他点的欧式距离
    d_arr = np.sqrt(x_delta ** 2 + y_delta ** 2)
    # 获取epsilon内的点
    d_arr[i] = eps + 1  # 将当前点的距离值大于eps，作用是在判断连接点时可将当前点过滤掉
    associate_indexes = np.where(d_arr <= eps)[0]
    visit_list[i] = 1
    if associate_indexes.shape[0] < min_pts: # 若当前点是核心点
        class_list[i] = -1  # 当前点置为噪声点
        continue
    # 当前点为核心点
    # 获取当前点的连结点
    search_list = associate_indexes.tolist()
    class_list[i] = label

    while len(search_list) > 0:
        j = search_list.pop(0) # 弹出第0个元素
        # 若当前点未被访问或者为噪声点
        if class_list[j] <= 0:
            class_list[j] = label
        if visit_list[j] == 0: # 若未被访问
            visit_list[j] = 1  # 更新为已被访问

            now_pt = X[j]
            x_delta = X[:, 0] - now_pt[0]
            y_delta = X[:, 1] - now_pt[1]
            # 计算当前点和其他点的欧式距离
            d_arr_now = np.sqrt(x_delta ** 2 + y_delta ** 2)
            inds = np.where(d_arr_now <= eps)[0]
            if inds.shape[0] >= min_pts:
                # 将未被访问的点作为搜索点添加至search_list后面
                not_visit_index = np.where(visit_list[inds] == 0)[0]  # visit_list[inds]中取==0的下标
                inds_ = inds[not_visit_index]
                search_list = list(set(search_list + inds_.tolist()))  # 更新搜索列表

                # 将噪声点的标签置为-1
                noise_pt_index = np.where((class_list[inds] <=0) & (visit_list[inds] == 1))[0]  # 噪声且已被访问的点
                # class_list[inds[noise_pt_index]] = label
                class_list[inds[noise_pt_index]] = label


plt.scatter(X[:, 0], X[:, 1], c=class_list)
plt.savefig("./example.png", dpi=300)
plt.show()









