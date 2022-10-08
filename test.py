# 1、PIL.Image转换成OpenCV格式：
import cv2
from PIL import Image
import numpy as np

path = 'D:/Workspace/Class.jpg'
img = Image.open(path).convert("RGB")  # .convert("RGB")可不要，默认打开就是RGB
img.show()

# 转opencv
# img = cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)
img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
cv2.imshow("OpenCV", img)
cv2.waitKey()