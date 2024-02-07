import cv2
import math
import numpy as np

def X_Y_cood(hough_lines, max_y):
    points = []

    for line in hough_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + (max_y + 100) * (-b))
        y1 = int(y0 + (max_y + 100) * (a))
        start = (x1, y1)

        x2 = int(x0 - (max_y + 100) * (-b))
        y2 = int(y0 - (max_y + 100) * (a))
        end = (x2, y2)

        points.append((start, end))

    return points

def remove_duplicate_lines(sorted_points):
    last_x1 = 0
    diff_points = []
    for point in sorted_points:
        ((x1, y1), (x2, y2)) = point
        if last_x1 == 0 or abs(last_x1 - x1) >= 25:
            diff_points.append(point)
            last_x1 = x1

    return diff_points


def ReduceLine(points, y_max):
    shortened_points = []

    for ((x1, y1), (x2, y2)) in points:
        try:
            m = (y2 - y1) / (x2 - x1)
        except ZeroDivisionError:
            m = float('inf') if y2 > y1 else -float('inf')
        
        if m == float('inf') or m == -float('inf'):
            shortened_points.append(((x1, y_max), (x1, 0)))
        else:
            new_x1 = int(((y_max - y1) / m) + x1)
            start_point = (new_x1, y_max)
            new_x2 = int(((0 - y1) / m) + x1)
            end_point = (new_x2, 0)
            shortened_points.append((start_point, end_point))

    return shortened_points


img = cv2.imread('/home/shourya/Downloads/BookCount_1.jpeg')
height, width, _ = img.shape
blur = cv2.GaussianBlur(img, (15, 15), 0)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(gray, 50, 70)

vertical_kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
], dtype=np.int32)

img_erosion = cv2.filter2D(edge, -1, vertical_kernel)
cv2.imwrite('eroded.jpeg', img_erosion)
lines = cv2.HoughLines(img_erosion, 1, np.pi / 180, 125)  

points = X_Y_cood(lines, height)
points.sort(key=lambda val: val[0][0])
diff_points = remove_duplicate_lines(points)
final_points = ReduceLine(diff_points, height)

for point in final_points:
    ((x1, y1), (x2, y2)) = point
    img = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

cv2.imwrite('final.jpeg', img)
