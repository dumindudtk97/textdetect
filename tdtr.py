from pkgutil import extend_path
import pytesseract
import cv2
import cv2
import imutils.object_detection 
import numpy as np
import time
from nonmax import non_max_suppression
from tkinter import *

root = Tk()
tktxt = Text(root)
tktxt.insert(INSERT, "Here is your text :\n")


net = cv2.dnn.readNet("frozen_east_text_detection.pb") # neural network file

def text_detector(image):
    orig = image
    (H,W) = image.shape[:2]

    (newW,newH) = (320,320)
    rW = W / float(newW)
    rH = H / float(newH)

    image = cv2.resize(image,(newW,newH))       # process images from stream
    (H,W) = image.shape[:2]

    layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]


    blob = cv2.dnn.blobFromImage(image,1.0,(W,H),(128.68,116.78,103.94),swapRB=True,crop=False)

    net.setInput(blob)
    (scores,geometry) = net.forward(layerNames)

    (numRows,numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0,numRows):
        scoresData = scores[0,0,y]
        xData0 = geometry[0,0,y]
        xData1 = geometry[0,1,y]
        xData2 = geometry[0,2,y]
        xData3 = geometry[0,3,y]
        anglesData = geometry[0,4,y]

        #loop over number of columns
        for x in range(0,numCols):
            # if our score does not have sufficient probability, ignore
            if scoresData[x] < 0.5:
                continue
            
            # comput the offset factor as our resulting feature maps with
            # be 4x smaller than input image
            (offsetX,offsetY) = (x*4.0,y*4.0)

            # extact rotation and angle
            # compute sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use geometry to comput bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY + (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX,startY,endX,endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects),probs=confidences)

    for ( startX, startY ,endX,endY) in boxes :
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY*rH)
        boundary = 20

        text = orig[startY-boundary:endY+boundary,startX - boundary:endX+boundary]
        text = cv2.cvtColor(text.astype(np.uint8),cv2.COLOR_BGR2GRAY)
        textRecognized = pytesseract.image_to_string(text)
        print("rec text : ")
        print(textRecognized)
        print("rec text end")
        tktxt.insert(INSERT,textRecognized)
        cv2.rectangle(orig,(startX-5,startY-5),(endX+5,endY+5),(0,255,0),3)
        orig = cv2.putText(orig,textRecognized,(endX,endY+5),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2,cv2.LINE_4)
    return orig

image0 = cv2.imread('ab.jpg')
img = image0

array = [image0]

for img in array:
    image = cv2.resize(img,(640,320),interpolation=cv2.INTER_AREA )
    orig = cv2.resize(img,(640,320),interpolation=cv2.INTER_AREA)
    textDetected = text_detector(image)
    cv2.imshow("orig image",orig)
    cv2.imshow("Text detected",textDetected)
    time.sleep(6)
    k = cv2.waitKey(30)
    if k == 27:
        break


cv2.destroyAllWindows()

tktxt.pack()

root.mainloop()