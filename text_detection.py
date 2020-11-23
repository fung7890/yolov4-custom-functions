from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2

# used for single image text detection, will get all text and return array of 'words' detected 

# NEED TO REMAKE OPEN CV FROM SOURCE TO USE GPU FOR INFERENCE 
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

def crop_object(img, startX, startY, endX, endY, count):
    cropped_img = img[int(startY):int(endY), int(startX):int(endX)]
    cv2.imshow("Cropped Image", cropped_img)
    cv2.imwrite("cropped%s.jpg" % count, cropped_img)


def text_detector(image):
    # resize for EAST usage
    image = cv2.resize(image, (640, 320), interpolation=cv2.INTER_AREA)
    orig = image
    
    # could be optimized later, leave out extra copy
    # orig_for_crop = image.copy()

    (H, W) = image.shape[:2]

    (newW, newH) = (640, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)

    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):

        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.5:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            padding = 5
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))+padding
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))+padding
            startX = int(endX - w)-padding
            startY = int(endY - h)-padding

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)
    # count = 0
    cropped_imgs = []
    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # crop_object(orig, startX, startY, endX, endY, count)
        # count += 1
        
        cropped_img = orig[int(startY):int(endY), int(startX):int(endX)]
        cropped_imgs.append(cropped_img)
        # draw the bounding box on the image
        # cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 3)

    return cropped_imgs


# image = cv2.imread('container3.jpg')
# image2 = cv2.imread('one.jpg')
# image3 = cv2.imread('crime.jpg')
# image4 = cv2.imread('hands.jpg')
# image5 = cv2.imread('beautiful.jpg')
# image6 = cv2.imread('everyday.jpg')

# array = [image]  # ,image2,image3,image4,image5,image6]

# for i in range(0, 1):
#     for img in array:
#         imageO = cv2.resize(img, (640, 320), interpolation=cv2.INTER_AREA)
#         imageX = imageO
#         orig = text_detector(imageO)
#         cv2.imshow("Text Detection", orig)
#         cv2.imwrite("container1text.jpg", orig)
#         k = cv2.waitKey(30) & 0xff
#         if k == 27:
#             break
# cv2.destroyAllWindows()
