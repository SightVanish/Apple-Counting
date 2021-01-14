from cv2 import cv2
import math
from hough import Hough_transform
from canny import Canny

Path = "picture_source/input"
# Path = "picture_source/picture"
Save_Path = "picture_result/"
name = '.jpg'
Reduced_ratio = 2
Gaussian_kernel_size = 3
HT_high_threshold = 32
HT_low_threshold = 8
# lower for more details
Hough_transform_step = 8
Hough_transform_threshold = 125
# higher may detect nothing

if __name__ == '__main__':
    img_gray = cv2.imread(Path + name, cv2.IMREAD_GRAYSCALE)
    img_RGB = cv2.imread(Path + name)
    y, x = img_gray.shape[0:2]
    img_gray = cv2.resize(img_gray, (int(x / Reduced_ratio), int(y / Reduced_ratio)))
    img_RGB = cv2.resize(img_RGB, (int(x / Reduced_ratio), int(y / Reduced_ratio)))
    # start running canny edge detector
    print ('===== Run Canny =====')
    canny = Canny(Gaussian_kernel_size, img_gray, HT_high_threshold, HT_low_threshold)
    canny.canny_algorithm()
    # save the result of canny
    cv2.imwrite(Save_Path + "canny_result" + name, canny.img)
    
    # start running hough transform
    print ('===== Run Hough Transform =====')
    Hough = Hough_transform(canny.img, canny.angle, Hough_transform_step, Hough_transform_threshold)
    circles = Hough.Calculate()
    # if not ((canny.img == Hough.img).all()):
    #     cv2.imwrite(Save_Path + "hough_result" + name, Hough.img)
    for circle in circles:
        cv2.circle(img_RGB, (math.ceil(circle[0]), math.ceil(circle[1])), math.ceil(circle[2]), (132, 135, 239), 2)
    print('number of circles:', len(circles))
    # save the final result - origin image with found circles marked out
    cv2.imwrite(Save_Path + "final_result" + name, img_RGB)
    print ('===== Done =====')
