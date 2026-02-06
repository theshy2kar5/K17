import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

M = 1024
N = 768

img = np.zeros((N,M), dtype=np.uint8)

cv.line(img, (0,0), (M-1, N-1), 255)

cv.imshow('image', img )
if cv.waitKey(0) == 27:
    cv.destroyAllWindows()

#Sử dụng CV vẽ một đường tròn trùng tâm với tâm của ảnh, có bán kính là 100, màu trắng, độ dày 2 pixel.