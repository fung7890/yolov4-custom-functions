import os
import cv2
import random
import numpy as np
import tensorflow as tf
import pytesseract
from core.utils import read_class_names
from core.config import cfg
# for EAST text detection
from text_detection import text_detector
from serial_number_recognizer import ocr_for_crop
# for CRAFT text detection
# from craft.text_detection import text_detector
# test
# function to count objects, can return total classes or count per class


def count_objects(data, by_class=False, allowed_classes=list(read_class_names(cfg.YOLO.CLASSES).values())):
    boxes, scores, classes, num_objects = data

    # create dictionary to hold count of objects
    counts = dict()

    # if by_class = True then count objects per class
    if by_class:
        class_names = read_class_names(cfg.YOLO.CLASSES)

        # loop through total number of objects found
        for i in range(num_objects):
            # grab class index and convert into corresponding class name
            class_index = int(classes[i])
            class_name = class_names[class_index]
            if class_name in allowed_classes:
                counts[class_name] = counts.get(class_name, 0) + 1
            else:
                continue

    # else count total objects found
    else:
        counts['total object'] = num_objects

    return counts

# function for cropping each detection and saving as new image

# PLACE HERE - crop container, EAST text detector on crop, crop text, run tesseract OCR  
def crop_objects(img, data, path, allowed_classes):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    # create dictionary to hold count of objects for image name
    counts = dict()
    for i in range(num_objects):
        # get count of class for part of image name
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name in allowed_classes:
            counts[class_name] = counts.get(class_name, 0) + 1
            # get box coords
            xmin, ymin, xmax, ymax = boxes[i]
            # crop detection from image (take an additional 5 pixels around all edges)
            # cropped_img = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
            
            # first crop, for container
            # can add padding but will need to do a min 0, max (length/width) - causes issues with cropping if not 
            cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]

            # # construct image name and join it to path for saving crop properly
            # img_name = class_name + '_' + str(counts[class_name]) + '.png'
            # img_path = os.path.join(path, img_name)
            # cv2.imwrite(img_path, cropped_img)

            # using EAST - uncomment line below and import for EAST text detection algorithm
            text_crop = text_detector(cropped_img)

            # using CRAFT 
            # text_crop = text_detector(cropped_img, i)

            count = 0
            for text_cropped in text_crop:
                try:
                    # construct image name and join it to path for saving crop properly
                    img_name = class_name + '_' + str(counts[class_name]) + str(count) + '.png'
                    txt_name = class_name + '_' + str(counts[class_name]) + str(count) + '.txt'
                    img_path = os.path.join(path, img_name)
                    txt_path = os.path.join(path, txt_name)
                    cv2.imwrite(img_path , text_cropped)
                    count += 1

                    try:
                        final_text_from_crop = ocr_for_crop(text_cropped, txt_path) # HEREEEEEEEEEEEEEEEEEEEEEEEE
                        print(final_text_from_crop)
                    except:
                        print("error from text crop")
                except:
                    print("error caused by: ", text_cropped)

        else:
            continue


# function to run general Tesseract OCR on any detections
def ocr(img, data):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    for i in range(num_objects):
        # get class name for detection
        class_index = int(classes[i])
        class_name = class_names[class_index]
        # separate coordinates from box
        xmin, ymin, xmax, ymax = boxes[i]
        # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
        box = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
        # grayscale region within bounding box
        gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
        # threshold the image using Otsus method to preprocess for tesseract
        thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # perform a median blur to smooth image slightly
        blur = cv2.medianBlur(thresh, 3)
        # resize image to double the original size as tesseract does better with certain text size
        blur = cv2.resize(blur, None, fx=2, fy=2,
                          interpolation=cv2.INTER_CUBIC)
        # run tesseract and convert image text to string
        try:
            text = pytesseract.image_to_string(blur, config='--psm 11 --oem 3')
            print("Class: {}, Text Extracted: {}".format(class_name, text))
        except:
            text = None
