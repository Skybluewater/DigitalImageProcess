import numpy as np
import PIL.Image as image
from sklearn import metrics
import cv2
from sklearn.cluster import SpectralClustering
from matplotlib import pyplot as plt


class kmeans_new:

    def load_data(self, file_path):
        f = open(file_path, "rb")  # 以二进制的方式打开图像文件
        data = []
        im = image.open(f)  # 导入图片
        m, n = im.size  # 得到图片的大小
        for i in range(n):
            for j in range(m):
                tmp = []
                x, y, z = im.getpixel((j, i))
                tmp.append(x / 256.0)
                tmp.append(y / 256.0)
                tmp.append(z / 256.0)
                data.append(tmp)
        f.close()
        return np.mat(data), m, n

    def process(self, file_path):
        img, row, col = self.load_data(file_path)
        out_list = []
        for index, gamma in enumerate((0.01, 0.1, 1, 10)):
            for index, k in enumerate((3, 4, 5, 6)):
                y_pred = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(img)
                score = metrics.calinski_harabaz_score(img, y_pred)
                out_list.append(score)
                print("Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", k, "score:", score)
        index = out_list.index(max(out_list))
        k = int(index // 4)
        x = int(index % 4)
        k, x = self.get_attributes(k, x)
        y_pred = SpectralClustering(n_clusters=x, gamma=k).fit_predict(img)
        img_a = np.zeros((col, row, 3))
        mask_all = np.zeros((col, row), dtype='uint8')
        for a in range(col):
            for b in range(row):
                mask_all[a][b] = np.uint(y_pred[a * row + b] * 256 / x)
                img_a[a][b][:] = img[a * row + b][:]
        cv2.imwrite("senior_segment_image.jpg", mask_all)
        return img_a, mask_all

    def get_attributes(self, k, x):
        if x == 0:
            x = 3
        elif x == 1:
            x = 4
        elif x == 2:
            x = 5
        else:
            x = 6

        if k == 0:
            k = 0.01
        elif k == 1:
            k = 0.1
        elif k == 2:
            k = 1
        else:
            k = 10
        return k, x


class MRF:
    def load_data(self, file_path):
        img = cv2.imread(file_path)
        return img

    def process(self, file_path, k):
        img = self.load_data(file_path=file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图片二值化，彩色图片该方法无法做分割
        img = gray
        img_double = np.array(img, dtype=np.float64)
        cluster_num = k
        max_iter = 200
        label = np.random.randint(1, cluster_num + 1, size=img_double.shape)
        iter = 0
        f_u = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]).reshape(3, 3)
        f_d = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]).reshape(3, 3)
        f_l = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]).reshape(3, 3)
        f_r = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]).reshape(3, 3)
        f_ul = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(3, 3)
        f_ur = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]).reshape(3, 3)
        f_dl = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]).reshape(3, 3)
        f_dr = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(3, 3)

        while iter < max_iter:
            iter = iter + 1
            label_u = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_u)
            label_d = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_d)
            label_l = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_l)
            label_r = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_r)
            label_ul = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_ul)
            label_ur = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_ur)
            label_dl = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_dl)
            label_dr = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_dr)
            m, n = label.shape
            p_c = np.zeros((cluster_num, m, n))
            for i in range(cluster_num):
                label_i = (i + 1) * np.ones((m, n))
                u_T = 1 * np.logical_not(label_i - label_u)
                d_T = 1 * np.logical_not(label_i - label_d)
                l_T = 1 * np.logical_not(label_i - label_l)
                r_T = 1 * np.logical_not(label_i - label_r)
                ul_T = 1 * np.logical_not(label_i - label_ul)
                ur_T = 1 * np.logical_not(label_i - label_ur)
                dl_T = 1 * np.logical_not(label_i - label_dl)
                dr_T = 1 * np.logical_not(label_i - label_dr)
                temp = u_T + d_T + l_T + r_T + ul_T + ur_T + dl_T + dr_T
                p_c[i, :] = (1.0 / 8) * temp
            p_c[p_c == 0] = 0.001
            mu = np.zeros((1, cluster_num))
            sigma = np.zeros((1, cluster_num))
            for i in range(cluster_num):
                index = np.where(label == (i + 1))
                data_c = img[index]
                mu[0, i] = np.mean(data_c)
                sigma[0, i] = np.var(data_c)
            p_sc = np.zeros((cluster_num, m, n))
            one_a = np.ones((m, n))
            for j in range(cluster_num):
                MU = mu[0, j] * one_a
                p_sc[j, :] = (1.0 / np.sqrt(2 * np.pi * sigma[0, j])) * np.exp(
                    -1. * ((img - MU) ** 2) / (2 * sigma[0, j]))
            X_out = np.log(p_c) + np.log(p_sc)
            label_c = X_out.reshape(cluster_num, m * n)
            label_c_t = label_c.T
            label_m = np.argmax(label_c_t, axis=1)
            label_m = label_m + np.ones(label_m.shape)  # 由于上一步返回的是index下标，与label其实就差1，因此加上一个ones矩阵即可
            label = label_m.reshape(m, n)
        label = label - np.ones(label.shape)  # 为了出现0
        lable_w = np.uint8(label * 256 / cluster_num)  # 此处做法只能显示两类，一类用0表示另一类用255表示
        cv2.imwrite('label.jpg', lable_w)
        return lable_w


class kmeans:
    def load_data(self, file_path):
        img = cv2.imread(file_path, 1)
        return img

    def classifer(self, data, center: np.ndarray):
        x, y, z = data.shape
        pixls_labels = np.zeros((x, y))
        pixl_distance_t = []
        for i in range(x):
            for j in range(y):
                for k in range(center.__len__()):
                    distance_t = np.sum(abs((data[i, j]).astype(int) - center[k].astype(int)) ** 2)
                    pixl_distance_t.append(distance_t)
                pixls_labels[i, j] = int(pixl_distance_t.index(min(pixl_distance_t)))
                pixl_distance_t = []
        return pixls_labels

    def initCentroids(self, data, k):
        centroids = np.zeros((k, 3))
        plt.title("click " + str(k) + " times to choose the inital centroids")
        plt.imshow(data, cmap="gray")
        pos = plt.ginput(k)
        plt.close()
        print(pos)
        for i in range(k):
            centroids[i, :] = data[int(pos[i][0]), int(pos[i][1]), :]
        return centroids

    def kmeans(self, data: np.ndarray, k):
        centroids = self.initCentroids(data, k)
        clusterData = self.classifer(data, centroids)
        threshold = 0
        while True:
            pre_centroids = centroids.copy()
            for i in range(k):
                temp = np.where(clusterData == i)
                centroids[i] = sum(data[temp].astype(int)) / len(data[temp])
            clusterData = self.classifer(data, centroids)
            loss = np.sum((centroids - pre_centroids) ** 2)
            print("Loss:" + str(loss))
            if loss <= threshold:
                break
        return clusterData

    def process(self, file_path, k):
        img = self.load_data(file_path)
        clusterData = self.kmeans(img, k)
        cv2.imwrite("clustered_image.jpg", np.uint8(clusterData * 256 / k))
        return clusterData


file_path = input('Please enter the name of the jpg file: ')

Kmeans = kmeans_new()

img_a, mask_all = Kmeans.process(file_path)

k = input('Please enter the number of cluster\'s number: ')

Mrf = MRF()

lable_w = Mrf.process(file_path, int(k))

kmeans = kmeans()

clusterData = kmeans.process(file_path, int(k))

plt.subplot(2, 2, 1)
plt.title("Original")
plt.imshow(img_a, cmap="gray")
plt.subplot(2, 2, 2)
plt.title("Spectral Clustering")
plt.imshow(mask_all, cmap="gray")
plt.subplot(2, 2, 3)
plt.title("MRF Clustering")
plt.imshow(lable_w, cmap="gray")
plt.subplot(2, 2, 4)
plt.title("Kmeans Clustering")
plt.imshow(clusterData / int(k), cmap="gray")
plt.show()
