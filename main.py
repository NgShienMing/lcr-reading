import os
import cv2
import pandas as pd
from ssocr_module import SSOCR

# path to the video file
video_path = 'asset/sample.mp4'

# open the video file
video = cv2.VideoCapture(video_path)
ssocr = SSOCR()
capacitance = []
units = []

while True:
    # read a frame from the video
    ret, frame = video.read()

    # if the frame was not successfully read, exit the loop
    if not ret:
        break

    cropped = frame[460:520, 185:380]
    number_region = cropped[:, 0:175]
    unit_region = cropped[40:60, 175:186]

    number, digit_img = ssocr.run_digit(number_region)
    unit = ssocr.run_unit(unit_region)
    unit = unit + 'F'

    capacitance.append(number)
    units.append(unit)

    # cv2.imshow('Number', digit_img)
    # cv2.waitKey(0)

    print(f'{number} {unit}')

# create folder if not exist
if not os.path.exists('runs'):
    os.makedirs('runs')

# count number of files in runs folder
files = os.listdir('runs')
index = len(files) + 1
filename = f'runs/capacitance_{index}.csv'

# write the capacitance values and units to csv using pandas
df = pd.DataFrame({'Capacitance': capacitance, 'Unit': units})
df.to_csv(filename, index=False)

# release the video file and close the window
video.release()
cv2.destroyAllWindows()
