import cv2
import numpy as  np

def clamp(pv):
    if pv > 255:
        return 255
    if pv < 0:
        return 0
    else:
        return pv

def gaussian_noise(image):
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)
            b = image[row, col, 0]  # blue
            g = image[row, col, 1]  # green
            r = image[row, col, 2]  # red
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    cv2.imshow("noise image",   image)


img=cv2.imread('input.jpg',1) #读取图片
cv2.imshow("input image", img)

# 高斯模糊抑制高斯噪声
dst = cv2.GaussianBlur(img, (43,43), 0)
cv2.imshow("Gaussian Blur", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #将图片变为灰度图片
kernel=np.ones((3,3),np.uint8) #进行腐蚀膨胀操作
erosion=cv2.erode(gray,kernel,iterations=15) 
dilation=cv2.dilate(erosion,kernel,iterations=25) 
ret, thresh = cv2.threshold(dilation, 175, 255, cv2.THRESH_BINARY) # 阈值处理 二值化
thresh1 = cv2.GaussianBlur(thresh,(3,3),0)# 高斯滤波
contours,hirearchy=cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)# 找连通域  

area=[] 
contours1=[]  
for i in contours:
     if cv2.contourArea(i)>30:  #去除面积小的 连通域
        contours1.append(i)
print(len(contours1)-1) #计算连通域个数
draw=cv2.drawContours(img,contours1,-1,(0,255,0),1) #描绘连通域

for i,j in zip(contours1,range(len(contours1))):
    M = cv2.moments(i)
    cX=int(M["m10"]/M["m00"])
    cY=int(M["m01"]/M["m00"])
    draw1=cv2.putText(draw, str(j), (cX, cY), 1,1, (255, 0, 255), 1) #在中心坐标点上描绘数字

cv2.imshow("draw",draw1)
cv2.imshow("thresh1",thresh1)
cv2.waitKey(0)
