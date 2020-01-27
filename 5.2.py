import numpy as np
import PIL.Image as image
from sklearn import metrics
import cv2
from sklearn.cluster import SpectralClustering
from matplotlib import pyplot as plt


def load_data(file_path):
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


img, row, col = load_data('b.jpg')

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
y_pred = SpectralClustering(n_clusters=x, gamma=k).fit_predict(img)
img_a = np.zeros((col, row, 3))
mask_all = np.zeros((col, row), dtype='uint8')
for a in range(col):
    for b in range(row):
        mask_all[a][b] = np.uint(y_pred[a * row + b] * 256 / x)
        img_a[a][b][:] = img[a * row + b][:]

plt.subplot(2, 1, 1)
plt.title("Original")
plt.imshow(mask_all, cmap="gray")
plt.subplot(2, 1, 2)
plt.title("Segmented")
plt.imshow(img_a, cmap="binary")
plt.show()
cv2.imwrite("senior_segment_image.jpg", mask_all)
