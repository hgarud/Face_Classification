import glob
import cv2
import linecache
import shutil
import random

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def test_loop_condition(iou):
    if iou < 0.3:
        return False
    else: return True

dir = "Data/FDDB-folds/"
image_dir_in = "Data/originalPics/"
# image_dir_out = "/home/hrishi/1Hrishi/ECE763_Comp_Vision/data/posImages_10x10/"
image_dir_out = "Data/posImages_gray/"
# neg_image_dir_out = "/home/hrishi/1Hrishi/ECE763_Comp_Vision/data/negImages/"
neg_image_dir_out = "Data/more_negImages/"
im_format = ".jpg"
numb = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
count = 118
neg_image_size = 60
for num in numb:
    filename_ellipse = dir+"FDDB-fold-"+num+"-ellipseList.txt"
    filename_images = dir+"FDDB-fold-"+num+".txt"
    file_images = open(filename_images,'r')
    image_names = file_images.readlines()

    for image_name in image_names:
        with open(filename_ellipse,'r') as file_ellipse:
            for num, line in enumerate(file_ellipse, 1):
                if image_name in line:
                    num_face_line = linecache.getline(filename_ellipse, (num+1))
                    if num_face_line.strip() == "1":
                        ellipse_dim = (linecache.getline(filename_ellipse, (num+2))).split()
                        image = cv2.imread(image_dir_in+image_name.strip()+im_format)
                        x1_postiv = int(float(ellipse_dim[3])-float(ellipse_dim[1]))            #//*
                        x2_postiv = int(float(ellipse_dim[3])+float(ellipse_dim[1]))            #*Create bounding box coordinates for Face images
                        y1_postiv = int(float(ellipse_dim[4])-float(ellipse_dim[0]))            #*
                        y2_postiv = int(float(ellipse_dim[4])+float(ellipse_dim[0]))            #*//
                        cropped_im = image[y1_postiv:y2_postiv,x1_postiv:x2_postiv]
                        # cv2.imshow('img', cropped_im)
                        if (cropped_im.shape[0]) and (cropped_im.shape[1]):
                            count = count+1
                            cropped_im = cv2.resize(cropped_im, (60,60))
                            # cv2.imwrite(image_dir_out+str(count)+im_format, cropped_im)
                            gray_im = cv2.cvtColor(cropped_im, cv2.COLOR_RGB2GRAY )
                            cv2.imwrite(image_dir_out+str(count)+im_format, gray_im)
                        condition = True
                        loop_count = 0
                        while condition:                                                        ## While condition to find random seeds until the overlap is less than threshold
                            rand_seed_x = random.randint(0,image.shape[0]-60)                   ## Set random seed for Non-Face patch extraction
                            rand_seed_y = random.randint(0,image.shape[1]-60)
                            # print('Seeds:', rand_seed_x, rand_seed_y)
                            x2_neg = rand_seed_x+neg_image_size
                            y2_neg = rand_seed_y+neg_image_size
                            boxA = [x1_postiv,y1_postiv,x2_postiv,y2_postiv]
                            boxB = [rand_seed_x,x2_neg,rand_seed_y,y2_neg]
                            iou = bb_intersection_over_union(boxA,boxB)
                            loop_count = loop_count+1
                            condition = test_loop_condition(iou)                                ## Check overlap
                        if (neg_image.shape[0]) and (neg_image.shape[1]):
                            count = count+1
                            neg_image = cv2.cvtColor(neg_image, cv2.COLOR_RGB2GRAY)
                            neg_image = cv2.resize(neg_image, (60, 60))
                            cv2.imwrite(neg_image_dir_out+str(count)+im_format, neg_image)
                            # cv2.waitKey(0)



# print("Done :)")
