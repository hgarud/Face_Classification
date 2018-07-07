import numpy as np
import cv2
import os
import matplotlib.pyplot as plot
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

class FDDB_Data:
    def __init__(self):
        self.pos_im_dir = "/home/hrishi/1Hrishi/ECE763_Comp_Vision/data/posImages_gray/"
        self.neg_im_dir = "/home/hrishi/1Hrishi/ECE763_Comp_Vision/data/negImages/"
        self.pos_test_im_dir = "/home/hrishi/1Hrishi/ECE763_Comp_Vision/data/testImages/pos/"
        self.neg_test_im_dir = "/home/hrishi/1Hrishi/ECE763_Comp_Vision/data/testImages/neg/"

    def load(self, train):
        if train:
            pos_images = os.listdir(self.pos_im_dir)
            neg_images = os.listdir(self.neg_im_dir)

            pos_vector_space = []
            for pos_image in pos_images:
                image = cv2.imread(self.pos_im_dir+pos_image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (10,10))
                im_vector = image.flatten()
                pos_vector_space.append(im_vector)
            pos_vector_space = np.array(pos_vector_space)

            neg_vector_space = []
            for neg_image in neg_images:
                image = cv2.imread(self.neg_im_dir+neg_image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (10,10))
                im_vector = image.flatten()
                neg_vector_space.append(im_vector)
            neg_vector_space = np.array(neg_vector_space)

            return pos_vector_space, neg_vector_space
        else:
            pos_test_images = os.listdir(self.pos_test_im_dir)
            neg_test_images = os.listdir(self.neg_test_im_dir)
            pos_test_vector_space = []
            for pos_test_image in pos_test_images:
                image = cv2.imread(self.pos_test_im_dir+pos_test_image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (10,10))
                im_vector = image.flatten()
                pos_test_vector_space.append(im_vector)
            pos_test_vector_space = np.array(pos_test_vector_space)

            neg_test_vector_space = []
            for neg_test_image in neg_test_images:
                image = cv2.imread(self.neg_test_im_dir+neg_test_image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (10,10))
                im_vector = image.flatten()
                neg_test_vector_space.append(im_vector)
            neg_test_vector_space = np.array(neg_test_vector_space)

            return pos_test_vector_space, neg_test_vector_space
