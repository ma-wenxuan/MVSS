import cv2
import matplotlib.pyplot as plt

image = cv2.imread('/home/mwx/d/graspnet/scenes/scene_0113/kinect/rgb/0000.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()