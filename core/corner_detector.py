import cv2
import numpy as np


def corner_detector(input_img, k = 0.04, window_size = 5, threshold = 3500.00,lambda_minus_flag = False):
    corner_list = []
    output_img = cv2.cvtColor(input_img.copy(), cv2.COLOR_GRAY2RGB)

    offset = int(window_size / 2)
    y_range = input_img.shape[0] - offset
    x_range = input_img.shape[1] - offset

    dy, dx = np.gradient(input_img)
    Ixx = dx ** 2
    Ixy = dy * dx
    Iyy = dy ** 2

    for y in range(offset, y_range):
        for x in range(offset, x_range):

            # Values of sliding window
            start_y = y - offset
            end_y = y + offset + 1
            start_x = x - offset
            end_x = x + offset + 1

            # The variable names are representative to
            # the variable of the Harris corner equation
            windowIxx = Ixx[start_y: end_y, start_x: end_x]
            windowIxy = Ixy[start_y: end_y, start_x: end_x]
            windowIyy = Iyy[start_y: end_y, start_x: end_x]

            # Sum of squares of intensities of partial derevatives
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            # Calculate determinant and trace of the matrix
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            
            if  lambda_minus_flag:
                discriminant = (trace**2) - (4 * det)
                # Prevent negative square roots due to floating point inaccuracies
                if discriminant < 0:
                    discriminant = 0
                # Calculate minimum eigenvalue (lambda minus)
                r = (trace - np.sqrt(discriminant)) / 2.0
            
            else:#Harris
                
                r = det - k * (trace ** 2)
                
            if r > threshold:
                corner_list.append([x, y, r])
                output_img[y, x] = (0, 0, 255)

    return corner_list, output_img