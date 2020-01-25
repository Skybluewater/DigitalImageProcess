import numpy as np
import cv2
from matplotlib import pyplot as plt
import random


def load_data(file_path):
    img = cv2.imread(file_path, 1)
    return img


def classifer(data, center: np.ndarray):
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


def initCentroids(data, k):
    x, y, z = data.shape
    centroids = np.zeros((k, z))
    rand_col = random.sample(range(0, y), k)
    rand_row = random.sample(range(0, x), k)
    for i in range(k):
        centroids[i, :] = data[rand_row[i], rand_col[i], :]
    return centroids


def kmeans(data: np.ndarray, k):
    centroids = initCentroids(data, k)
    clusterData = classifer(data, centroids)
    while True:
        pre_centroids = centroids.copy()
        for i in range(k):
            temp = np.where(clusterData == i)
            centroids[i] = sum(data[temp].astype(int)) / len(data[temp])
        clusterData = classifer(data, centroids)
        loss = np.sum((centroids - pre_centroids) ** 2)
        print("Loss:" + str(loss))
        if loss <= 6:
            break
    return clusterData


img = load_data('b.jpg')
clusterData = kmeans(img, 3)
plt.subplot(2, 1, 1)
plt.title("original")
plt.imshow(img, cmap="binary")
plt.subplot(2, 1, 2)
plt.title("clustering result")
plt.imshow(clusterData / 3, cmap="gray")
plt.show()

cv2.waitKey(0)
cv2.imwrite("clustered_image.jpg", np.uint8(clusterData*256/3))
