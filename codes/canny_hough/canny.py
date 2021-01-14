from cv2 import cv2
import numpy as np

class Canny:
    def __init__(self, Gaussian_kernel_size, img, HT_high_threshold, HT_low_threshold):
        '''
        :param Gaussian_kernel_size: the size of Gaussian kernel to filter the noise
        :param img: the input image, will change to the result of canny edge detection
        :param HT_high_threshold: the high threshold in Hysteresis Thresholding
        :param HT_low_threshold: the low threshold in Hysteresis Thresholding
        '''
        self.Gaussian_kernel_size = Gaussian_kernel_size
        self.img = img
        self.y, self.x = img.shape[0:2]
        self.angle = np.zeros([self.y, self.x])
        self.img_origin = None
        self.x_kernel = np.array([[-1, 1]])
        self.y_kernel = np.array([[-1], [1]])
        self.HT_high_threshold = HT_high_threshold
        self.HT_low_threshold = HT_low_threshold

    def Get_gradient_img(self):
        '''
        calculate the intensity gradient of the image
        :return: the gradient image
        '''
        print ('calculating gradient image')
        
        new_img_x = np.zeros([self.y, self.x], dtype=np.float)
        new_img_y = np.zeros([self.y, self.x], dtype=np.float)
        for i in range(0, self.x):
            for j in range(0, self.y):
                if j == 0:
                    new_img_y[j][i] = 1
                else:
                    new_img_y[j][i] = np.sum(np.array([[self.img[j - 1][i]], [self.img[j][i]]]) * self.y_kernel)
                if i == 0:
                    new_img_x[j][i] = 1
                else:
                    new_img_x[j][i] = np.sum(np.array([self.img[j][i - 1], self.img[j][i]]) * self.x_kernel)

        gradient_img, self.angle = cv2.cartToPolar(new_img_x, new_img_y)
        self.angle = np.tan(self.angle)
        self.img = gradient_img.astype(np.uint8)
        return self.img

    def Non_maximum_suppression (self):
        '''
        Do Non-maximum Suppression on the gradient image
        Decide the direction of the gradient by the value and the sign of tangent
        :return: the result image of Non-maximum Suppression
        '''
        print ('doing Non-maximum Suppression')
        
        result = np.zeros([self.y, self.x])
        for i in range(1, self.y - 1):
            for j in range(1, self.x - 1):
                if abs(self.img[i][j]) <= 4:
                    result[i][j] = 0
                    continue
                elif abs(self.angle[i][j]) > 1:
                    gradient2 = self.img[i - 1][j]
                    gradient4 = self.img[i + 1][j]
                    # g1 g2
                    #    C
                    #    g4 g3
                    if self.angle[i][j] > 0:
                        gradient1 = self.img[i - 1][j - 1]
                        gradient3 = self.img[i + 1][j + 1]
                    #    g2 g1
                    #    C
                    # g3 g4
                    else:
                        gradient1 = self.img[i - 1][j + 1]
                        gradient3 = self.img[i + 1][j - 1]
                else:
                    gradient2 = self.img[i][j - 1]
                    gradient4 = self.img[i][j + 1]
                    # g1
                    # g2 C g4
                    #      g3
                    if self.angle[i][j] > 0:
                        gradient1 = self.img[i - 1][j - 1]
                        gradient3 = self.img[i + 1][j + 1]
                    #      g3
                    # g2 C g4
                    # g1
                    else:
                        gradient3 = self.img[i - 1][j + 1]
                        gradient1 = self.img[i + 1][j - 1]

                temp1 = abs(self.angle[i][j]) * gradient1 + (1 - abs(self.angle[i][j])) * gradient2
                temp2 = abs(self.angle[i][j]) * gradient3 + (1 - abs(self.angle[i][j])) * gradient4
                if self.img[i][j] >= temp1 and self.img[i][j] >= temp2:
                    result[i][j] = self.img[i][j]
                else:
                    result[i][j] = 0
        self.img = result
        return self.img

    def Hysteresis_thresholding(self):
        '''
        Apply Hysteresis Thresholding on the result image of Non-maximum Suppression
        Strengthen the weak edges between thresholds by strong edges,
        extend along the vertical direction of gadient, set the value to high threshold
        :return: the result image of Hysteresis Thresholding
        '''
        print ('Applying Hysteresis Thresholding')
        
        for i in range(1, self.y - 1):
            for j in range(1, self.x - 1):
                if self.img[i][j] >= self.HT_high_threshold:
                    if abs(self.angle[i][j]) < 1:
                        if self.img_origin[i - 1][j] > self.HT_low_threshold:
                            self.img[i - 1][j] = self.HT_high_threshold
                        if self.img_origin[i + 1][j] > self.HT_low_threshold:
                            self.img[i + 1][j] = self.HT_high_threshold
                        # g1 g2
                        #    C
                        #    g4 g3
                        if self.angle[i][j] < 0:
                            if self.img_origin[i - 1][j - 1] > self.HT_low_threshold:
                                self.img[i - 1][j - 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j + 1] > self.HT_low_threshold:
                                self.img[i + 1][j + 1] = self.HT_high_threshold
                        #    g2 g1
                        #    C
                        # g3 g4
                        else:
                            if self.img_origin[i - 1][j + 1] > self.HT_low_threshold:
                                self.img[i - 1][j + 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j - 1] > self.HT_low_threshold:
                                self.img[i + 1][j - 1] = self.HT_high_threshold
                    else:
                        if self.img_origin[i][j - 1] > self.HT_low_threshold:
                            self.img[i][j - 1] = self.HT_high_threshold
                        if self.img_origin[i][j + 1] > self.HT_low_threshold:
                            self.img[i][j + 1] = self.HT_high_threshold
                        # g1
                        # g2 C g4
                        #      g3
                        if self.angle[i][j] < 0:
                            if self.img_origin[i - 1][j - 1] > self.HT_low_threshold:
                                self.img[i - 1][j - 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j + 1] > self.HT_low_threshold:
                                self.img[i + 1][j + 1] = self.HT_high_threshold
                        #      g3
                        # g2 C g4
                        # g1
                        else:
                            if self.img_origin[i - 1][j + 1] > self.HT_low_threshold:
                                self.img[i + 1][j - 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j - 1] > self.HT_low_threshold:
                                self.img[i + 1][j - 1] = self.HT_high_threshold
        return self.img

    def canny_algorithm(self):
        '''
        execute Canny in the sequence of Gaussian Blur, gradient image,
        Non-maximum Suppression and Hysteresis Thresholding
        :return: the result image of Canny Edge Detection
        '''
        self.img = cv2.GaussianBlur(self.img, (self.Gaussian_kernel_size, self.Gaussian_kernel_size), 0)
        self.Get_gradient_img()
        self.img_origin = self.img.copy()
        self.Non_maximum_suppression()
        self.Hysteresis_thresholding()
        return self.img
