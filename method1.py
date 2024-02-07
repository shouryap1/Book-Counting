import cv2
import numpy as np

image = cv2.imread('/content/BookCount_2.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 30, 200)
contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.imwrite('/content/canny_edges.jpg', edged)

print("Number of Contours found = " + str(len(contours)))

values=450
filtered_contours = [contour for contour in contours if len(contour) > values]
cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 3)
file_str='/content/contours_above_test_'+str(values)+'.jpg'
cv2.imwrite(file_str, image)

print(len(filtered_contours))