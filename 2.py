import cv2
import numpy as np
import matplotlib.pyplot as plt


def warpPerspectiveMatrix(src: np.ndarray, dst: np.ndarray):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))  # A*warpMatrix=B
    B = np.zeros((2 * nums, 1))
    for i in range(0, nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1,
                       0, 0, 0,
                       -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]
        A[2 * i + 1, :] = [0, 0, 0,
                           A_i[0], A_i[1], 1,
                           -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]
    A = np.mat(A)
    warpMatrix = A.I * B
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix


image = cv2.imread('a.jpg', 0)
rows, cols = image.shape
src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])

plt.imshow(image, cmap="gray")
pos = plt.ginput(4)
plt.close()
print(pos)
dst_points = np.float32(pos)

projective_martix = warpPerspectiveMatrix(src_points, dst_points)

print(projective_martix)


def perspectiveImage(image, projective_martix):
    image_return = np.copy(image)
    m, n = image_return.shape
    mask_all = np.zeros((m, n), dtype='uint8')
    plt.imshow(mask_all, cmap="gray")
    for a in range(m):
        for b in range(n):
            y = projective_martix[0][0]*b + projective_martix[0][1]*a + projective_martix[0][2]
            x = projective_martix[1][0]*b + projective_martix[1][1]*a + projective_martix[1][2]
            k = projective_martix[2][0]*b + projective_martix[2][1]*a + projective_martix[2][2]
            mask_all[int(x/k)][int(y/k)] = image_return[a][b]
    return mask_all


projective_image1 = perspectiveImage(image, projective_martix)
projective_image2 = cv2.warpPerspective(image, projective_martix, (cols, rows))

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap="gray")
plt.subplot(2, 2, 2)
plt.title('Projective Image1')
plt.imshow(projective_image1, cmap="gray")
plt.subplot(2, 2, 3)
plt.title('Projective Image2')
plt.imshow(projective_image2, cmap="gray")
plt.show()
cv2.waitKey(0)
cv2.imwrite("Perspective_image1.jpg", projective_image2)
cv2.imwrite("Projective_image.jpg", projective_image1)
