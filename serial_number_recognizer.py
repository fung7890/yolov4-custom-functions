import cv2
import pytesseract

# specifiy tesseract path
pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'

def ocr_for_crop(img, path):
    print("start of ocr")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    gray = cv2.medianBlur(gray, 3)

    # perform otsu thresh (using binary inverse since opencv contours work better with white text)

    ret, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    threshS = cv2.resize(thresh, (960, 540))                    # Resize image

    # cv2.imshow("Otsu", threshS)
    # cv2.waitKey(0)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # apply dilation
    dilation = cv2.dilate(thresh, rect_kern, iterations=1)


    # Adding custom options
    custom_config = r'--oem 3 --psm 3'

    output = pytesseract.image_to_string(dilation, config=custom_config)
    if len(output) > 1:
        print(path)
        f = open(path, "w")
        f.write(output)
        f.close()
    print("output ", output)
    return output

