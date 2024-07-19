import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

class SSOCR:
    DIGITS_LOOKUP = {
        (1, 1, 1, 1, 1, 1, 0): 0,
        (1, 1, 0, 0, 0, 0, 0): 1,
        (1, 0, 1, 1, 0, 1, 1): 2,
        (1, 1, 1, 0, 0, 1, 1): 3,
        (1, 1, 0, 0, 1, 0, 1): 4,
        (0, 1, 1, 0, 1, 1, 1): 5,
        (0, 1, 1, 1, 1, 1, 1): 6,
        (1, 1, 0, 0, 0, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 0, 1, 1, 1): 9,
        (0, 0, 0, 0, 0, 1, 1): '-'
    }
    H_W_Ratio = 1.9
    THRESHOLD = 20
    arc_tan_theta = 6.0  # 数码管倾斜角度

    def __init__(self):
        pass

    def preprocess(self, img, show=False):
        # convert image to grayscale
        rows, cols, _ = img.shape
        M = np.float32([[1, -0.1, 0], [0, 1, 0], [0, 0, 1]])
        warped_img = cv2.warpPerspective(img, M, (int(cols*1)-10, int(rows*1)))
        gray_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)

        # apply GaussianBlur to smooth image
        blurred_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
        if show:
            cv2.imshow('gray_img', gray_img)
            cv2.imshow('blurred_img', blurred_img)

        return warped_img, gray_img, blurred_img

    def thresholding(self, img, threshold, method=1, show=False, kernel_size=(5, 5)):
        if method == 1:
            # 直方图局部均衡化
            clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
            img = clahe.apply(img)
            # 自适应阈值二值化
            dst = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 127, threshold)
            # 闭运算开运算
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
            dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
            dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
        elif method == 2:
            thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
            dst = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        if show:
            cv2.imshow('equlizeHist', img)
            cv2.imshow('threshold', dst)

        return dst

    def helper_extract(self, one_d_array, mode, threshold=20):
        res = []
        flag = 0
        temp = 0
        for i in range(len(one_d_array)):
            if one_d_array[i] < 10 * 255:
                if flag > threshold:
                    start = i - flag
                    end = i
                    temp = end
                    if end - start > 0:
                        if mode == 'h':
                            end = end + 1
                        elif mode == 'v':
                            end = end - 1
                        res.append((start, end))
                flag = 0
            else:
                flag += 1

        else:
            if flag > threshold:
                start = temp
                end = len(one_d_array)
                if end - start > 0:
                    if mode == 'h':
                        end = end + 1
                    elif mode == 'v':
                        end = end - 1
                    res.append((start, end))

        return res
    
    def find_digits_positions(self, img, reserved_threshold=10):
        digits_positions = []
        img_array = np.sum(img, axis=0)
        horizon_position = self.helper_extract(img_array, mode='h', threshold=reserved_threshold)
        img_array = np.sum(img, axis=1)
        vertical_position = self.helper_extract(img_array, mode='v', threshold=reserved_threshold)
        
        # make vertical_position has only one element
        if len(vertical_position) > 1:
            vertical_position = [(vertical_position[0][0], vertical_position[len(vertical_position) - 1][1])]
        for h in horizon_position:
            for v in vertical_position:
                digits_positions.append(list(zip(h, v)))
        assert len(digits_positions) > 0, "Failed to find digits's positions"

        return digits_positions

    def recognize_digits_line_method(self, digits_positions, output_img, input_img):
        digits = []
        for c in digits_positions:
            x0, y0 = c[0]
            x1, y1 = c[1]
            roi = input_img[y0:y1, x0:x1]
            h, w = roi.shape
            suppose_W = max(1, int(h / self.H_W_Ratio))

            # 消除无关符号干扰
            if x1 - x0 < 25 and cv2.countNonZero(roi) / ((y1 - y0) * (x1 - x0)) < 0.2:
                continue

            # 对1的情况单独识别
            if w < suppose_W / 2:
                x0 = max(x0 + w - suppose_W, 0)
                roi = input_img[y0:y1, x0:x1]
                w = roi.shape[1]

            center_y = h // 2
            quater_y_1 = h // 4
            quater_y_3 = quater_y_1 * 3
            center_x = w // 2
            line_width = 5  # line's width
            width = (max(int(w * 0.15), 1) + max(int(h * 0.15), 1)) // 2
            small_delta = int(h / self.arc_tan_theta) // 4
            segments = [
                ((w - 2 * width, quater_y_1 - line_width), (w, quater_y_1 + line_width)),
                ((w - 2 * width, quater_y_3 - line_width), (w, quater_y_3 + line_width)),
                ((center_x - line_width - small_delta, h - 2 * width), (center_x - small_delta + line_width, h)),
                ((0, quater_y_3 - line_width), (2 * width, quater_y_3 + line_width)),
                ((0, quater_y_1 - line_width), (2 * width, quater_y_1 + line_width)),
                ((center_x - line_width, 0), (center_x + line_width, 2 * width)),
                ((center_x - line_width, center_y - line_width), (center_x + line_width, center_y + line_width)),
            ]
            on = [0] * len(segments)

            for (i, ((xa, ya), (xb, yb))) in enumerate(segments):
                seg_roi = roi[ya:yb, xa:xb]
                total = cv2.countNonZero(seg_roi)
                area = (xb - xa) * (yb - ya) * 0.9
                if total / float(area) > 0.25:
                    on[i] = 1
            if tuple(on) in self.DIGITS_LOOKUP.keys():
                digit = self.DIGITS_LOOKUP[tuple(on)]
            else:
                digit = '*'

            digits.append(digit)

            # 小数点的识别
            if cv2.countNonZero(roi[h - int(4 * width / 4):h, w - int(4 * width / 4):w]) / (9. / 16 * width * width) > 0.5:
                digits.append('.')
                cv2.rectangle(output_img,
                            (x0 + w - int(4 * width / 4), y0 + h - int(4 * width / 4)),
                            (x1, y1), (255, 0, 0), 1)
                cv2.putText(output_img, 'dot',
                            (x0 + w - int(3 * width / 4), y0 + h - int(3 * width / 4) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1)

            cv2.rectangle(output_img, (x0, y0), (x1, y1), (255, 0, 0), 1)
            cv2.putText(output_img, str(digit), (x0 + 3, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1)
        
        return digits
    
    def recognize_unit(self, binary_img):
        # Find contours
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Assume the largest contour is the letter
        contour = max(contours, key=cv2.contourArea)
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        # Calculate aspect ratio
        aspect_ratio = float(w) / h

        row_sum = np.sum(binary_img, axis=1)
        first_non_zero = np.argmax(row_sum > 0)
        last_non_zero = len(row_sum) - np.argmax(row_sum[::-1] > 0)
        index_sum = first_non_zero + last_non_zero

        if aspect_ratio > 0.78 and index_sum > 20:
            return 'u'
        else:
            return 'n'
    
    def run_digit(self, img, show=False):
        warped_img, gray_img, blurred_img = self.preprocess(img, show=show)
        thresholded = self.thresholding(blurred_img, self.THRESHOLD, method=2, show=show)
        digits_positions = self.find_digits_positions(thresholded)
        digits = self.recognize_digits_line_method(digits_positions, warped_img, thresholded)
        number_str = ''.join(map(str, digits))

        # try: 
        #     number = float(number_str)
        # except:
        #     number = number_str
        return number_str, warped_img
    
    def run_unit(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresholded = cv2.threshold(gray_img, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        unit = self.recognize_unit(thresholded)
        return unit
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSOCR')
    parser.add_argument('image_path', help='path to image')
    parser.add_argument('-s', '--show_image', action='store_const', const=True, help='whether to show image')
    args = parser.parse_args()

    ssocr = SSOCR()
    img = cv2.imread(args.image_path)
    number, digit_img = ssocr.run_digit(img, show=args.show_image)
    print("Number: ", number)
    cv2.imshow('Output', digit_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()