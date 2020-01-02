# Created on Jan 1st 2020
# Author: Changzhi Cai
# Contact me: caichangzhi97@gmail.com

# import package
import numpy as np
import cv2

# set correct answer
ANSWER_KEY = {0:1,1:4,2:0,3:3,4:1}

# show the image
def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(100)
    cv2.destroyAllWindows() 
    
def order_points(pts):
    
    # totally 4 points
	rect = np.zeros((4,2), dtype = "float32")

    # point: 0 - left_up, 1 - right_up, 2 - right_down, 3 - left_down
    # calculate point 0 and point 2
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

    # calculate point 1 and point 3
	diff = np.diff(pts,axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def four_point_transform(image,pts):
    
    # get import points
	rect = order_points(pts)
	(tl,tr,br,bl) = rect

    # calculate the width and take the max value
	widthA = np.sqrt(((br[0]-bl[0])**2)+((br[1]-bl[1])**2))
	widthB = np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2))
	maxWidth = max(int(widthA),int(widthB))

    # calculate the height and take the max value
	heightA = np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))
	heightB = np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))
	maxHeight = max(int(heightA),int(heightB))

    # loaction after transformation
	dst = np.array([
		[0,0],
		[maxWidth-1,0],
		[maxWidth-1,maxHeight-1],
		[0,maxHeight-1]],dtype = "float32")

    # calculate transformed matrix
	M = cv2.getPerspectiveTransform(rect,dst)
	warped = cv2.warpPerspective(image,M,(maxWidth, maxHeight))

	return warped

def sort_contours(cnts,method = "left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts,boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts,boundingBoxes

# do the preprocess
image = cv2.imread("test_02.png")
contours_img = image.copy()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(5,5),0)
cv_show('blurred',blurred)
edged = cv2.Canny(blurred,75,200)
cv_show('edged',edged)

# edge detection
cnts = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
cv2.drawContours(contours_img,cnts,-1,(0,0,255),3)
cv_show('contours_img',contours_img)
docCnt = None

# ensure it is detected
if len(cnts)>0:
    
    # sorting by the contour
    cnts = sorted(cnts,key = cv2.contourArea,reverse = True)
    
    # go through each contour
    for c in cnts:
        
        # approximation
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.02*peri,True)
        
        # ready for perspective transformation
        if len(approx) == 4:
            docCnt = approx
            break

# do the perspective tranformation
warped = four_point_transform(gray,docCnt.reshape(4,2))
cv_show('warped',warped)

# Otsu's threshold process
thresh = cv2.threshold(warped,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
cv_show('thresh',thresh)
thresh_Contours = thresh.copy()

# find each circle contour
cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
cv2.drawContours(thresh_Contours,cnts,-1,(0,0,255),3)
cv_show('thresh_Contours',thresh_Contours)
questionCnts = []

# traverse
for c in cnts:
    
    # calculate proportion and area
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w/float(h)
    
    # set the standard according to the condition
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)
        
# sorting from up to down
questionCnts = sort_contours(questionCnts,method = "top-to-buttom")[0]
correct = 0

# for five options in one question
for (q,i) in enumerate(np.arange(0,len(questionCnts),5)):
    
    # sorting
    cnts = sort_contours(questionCnts[i:i+5])[0]
    bubbled = None
    
    # go through each result
    for (j,c) in enumerate(cnts):
        
        # use mask to determine the results
        mask = np.zeros(thresh.shape,dtype = "uint8")
        cv2.drawContours(mask,[c],-1,255,-1)
        cv_show('mask',mask)
        
        # determine whether select it by calculating the number on non-zero points
        mask = cv2.bitwise_and(thresh,thresh,mask = mask)
        total = cv2.countNonZero(mask)
        
        # use threshold to judge
        if bubbled is None or total > bubbled[0]:
            bubbled = (total,j)
            
    # compare with correct answer
    color = (0,0,255)
    k = ANSWER_KEY[q]
    
    # it is correct
    if k == ANSWER_KEY[1]:
        color = (0,255,0)
        correct += 1
        
    # draw it
    cv2.drawContours(warped,[cnts[k]],-1,color,3)
    
score = (correct/5.0)*100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(warped, "{:.2f}%".format(score),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
cv2.imshow("Original",image)
cv2.imshow("Exam",warped)
cv2.waitKey(0)