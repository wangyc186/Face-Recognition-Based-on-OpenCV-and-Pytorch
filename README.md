# Face-Recognition-Based-on-OpenCV-and-Pytorch
Use Pytorch Deep Learning  
Partial reference https://blog.csdn.net/qq_44707179/article/details/117135230  

You have to install OpenCV and Pytorch first,based on your OS!  

OpenCV  
'''sh
import cv2
CADES_PATH = 'opencv-4.5.2/data/haarcascades_cuda/haarcascade_frontalface_al
def face_detect(img_path) :
color = (0, 255, 0)
img_bgr = cv2. imread (img_path)
classifier = cv2. CascadeClassifier (CADES_PATH)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
facerects = classifier.detectMultiScale(img_gray)
if len(facerects) > 0: for rect in facerects:
x, y, w, h = rect
if w > 200:
cv2. rectangle(img_bgr, (x, y), (x +w, y + h), color, 2)
cv2. imwrite('detect.png', img_bgr)
if
_name_ == '_main_': face_detect( 'eg.jpg')
'''
