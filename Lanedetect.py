
# coding: utf-8

# In[2]:


import cv2
import numpy as np
#import matplotlib.pyplot as plt # to show the x and y axis so that we can get the coordinate of specific area of interest(def Region)

def canny(image):
	grayimage= cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)# converting color image to grayscale to reduce computation
	blurimage= cv2.GaussianBlur(grayimage,(5,5),0)# Using gaussianblur kernel of 5x5 to smoothen the image to reduce noise(Canny includes Gaussianblur
	cannyimage= cv2.Canny(blurimage,50,150)# we will take the input from blur image, we use canny edge detection to seperate the lane lines(image,lower threshold, upperthreshold)
	return cannyimage;

def Newhoughlines(image, Mainlines):
	lineimage= np.zeros_like(image)# Declaring array of zeros of same shape of the real image
	if Mainlines is not None:# Checking whether houghlines detected any lines, so puting condition whether is none or not 
		for line in Mainlines:
			x1,y1,x2,y2=line.reshape(4)# reshaping to one dimensinal array of 4 elements(before that it was 2D) 
			#print(x1,y1,x2,y2)
			cv2.line(lineimage,(x1,y1),(x2,y2),(255,255,),8)#Taking each line that iterates and draw it in lineimage (image,coordinates,color of theline, thickness in pixels)
	return lineimage
def Region(image):#Defining area of interest in the image
	height= image.shape[0]#this will be equal to the number of rows which will be equal to the max value of Y axis
	Triangle= np.array([[(200,height),(1100,height),(705,345)]])# Should be mention as array of polygons
	mask = np.zeros_like(image)#It will have same dimension as image but zero intensity which makes it black)
	cv2.fillPoly(mask,Triangle,255)#filling the mask with triangle using fillpoly function
	maskedimage = cv2.bitwise_and(image,mask)#isolated the required region and masked remaining region(Bitwise and operation between canny and masked image)
	return maskedimage;

def Newcoordinates(image,line_parameters):# to draw the line with avg slope and intercept taken from avgslopeintercept function
	slope, intercept = line_parameters
	y1= image.shape[0]# this will be no of rows, roughly 700
	#print(y1)
	y2= int(y1*(3/5))
	x1 = int((y1 - intercept)/slope)
	x2= int((y2 - intercept)/slope)
	return np.array([x1,y1,x2,y2])

def avgslopeintercept(image,Mainlines):# funtion to average out multiple lines(as a result of new houghline function ) to a single smooth line
	leftfit= []# have coordinates of avg lines in left
	rightfit= []#have coordinates of avg lines in right
	for line in Mainlines:# looping through every line like before
		x1,y1,x2,y2 = line.reshape(4)
		#print(x1, x2, y1, y2)
		parameters = np.polyfit((x1,x2),(y1,y2),1)# to find the slope of each line, using polyfit of degree'1' linear and fit this line in x1,y1,x2,y2 coordinates given as parameter
		#print(parameters)
		slope = parameters[0]#index 0 is slope
		intercept = parameters[1]#index 1 is intercept
		if slope < 0:# line which are having slope negative are on the left side and positive slope on right side(m=(y2-y1)/(x2-x1))
			leftfit.append((slope,intercept))
		else:
			rightfit.append((slope,intercept))
			#print(leftfit)
			#print(rightfit)
			leftfitaverage = np.average(leftfit, axis=0)#average out all value on leftside, operate vertical axis=0 to get the avg
			rightfitaverage= np.average(rightfit, axis=0)#avg out all value on right side
			leftline= Newcoordinates(image,leftfitaverage)#Drawing final left line 
			rightline= Newcoordinates(image,rightfitaverage)# drawing final right line
			return np.array([leftline,rightline])

#For an image
image= cv2.imread('test_image.jpg')
lane_image = np.copy(image)# copying to array format
cannyimage = canny(lane_image)
newimage= Region(cannyimage)
Mainlines= cv2.HoughLinesP(newimage,2,np.pi/180,60,np.array([]),minLineLength=30,maxLineGap=5)#Detect straightline to find lane using hough transformation(newimage, resolution of Hough accumulator array which contain bins(ro and theta)(2,np.pi/180) to find max vote, threshold of min number of votes(60) in bin,placeholder array nothing,Length of the line in pixel to be accepted, max gap which can be connected to a single line)
Lineavg= avgslopeintercept(lane_image,Mainlines)
lineimage= Newhoughlines(lane_image,Lineavg)
Imagemix= cv2.addWeighted(lane_image,0.8,lineimage,1,1)#multiplying lane_image intensity with added weight of 0.8 and Mainlines with 1 and then adding with lineimage, gamma argument=1
#plt.imshow(cannyimage)
#plt.show()
#cv2.imshow("xy",cannyimage)
cv2.imshow("Final",Imagemix)# It will show the image until we press a key
cv2.waitKey(0)

#For video
# capture = cv2.VideoCapture("video.mp4")
# while(capture.isOpened()):
# 	_, frame = capture.read()
# 	cannyimage = canny(frame)
# 	newimage= Region(cannyimage)
# 	Mainlines= cv2.HoughLinesP(newimage,2,np.pi/180,60,np.array([]),minLineLength=30,maxLineGap=5)
# 	Lineavg= avgslopeintercept(frame,Mainlines)
# 	lineimage= Newhoughlines(frame,Lineavg)
# 	Imagemix= cv2.addWeighted(frame,0.8,lineimage,1,1)
# 	cv2.imshow("Final",Imagemix)
# 	if cv2.waitKey(1) == ord('q'):#wait for 1 milli second between frames
# 		break
# capture.release()
# cv2.destroyAllWindows()

