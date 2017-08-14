import numpy as np
import cv2

cap=cv2.VideoCapture(0)
while True:
	ret, img=cap.read()
	hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	kernel=np.ones((5,5),np.uint8)
	Lower_green = np.array([0, 59, 90])
	Upper_green = np.array([3, 255, 255])
#	Lower_green = np.array([1, 100, 80])
#	Upper_green = np.array([2, 255, 255])
	mask=cv2.inRange(hsv,Lower_green,Upper_green)
	#mask = cv2.erode(mask, kernel, iterations=2)
	mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
	#mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
	mask = cv2.dilate(mask, kernel, iterations=1)
	res=cv2.bitwise_and(img,img,mask=mask)
	cnts,heir=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
	center = None

	if len(cnts) > 0:
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		if radius > 10:
			cv2.circle(img, (int(x), int(y)), int(radius),(0, 255, 255), 2)
			cv2.circle(img, center, 5, (0, 0, 255), -1)



	cv2.imshow("Frame", img)
	cv2.imshow("mask",mask)
	cv2.imshow("res",res)





	k=cv2.waitKey(1) & 0xFF
	if k==27:
		break
# Release the Camera and close AllWindows
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(10)
cv2.waitKey(10)
cv2.waitKey(10)
