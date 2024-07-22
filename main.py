import os
import cv2
import pandas as pd
from ssocr_module import SSOCR

# path to the video file
video_path = 'asset/experiment1_1_5.mp4'

# open the video file
video = cv2.VideoCapture(video_path)

# frame skipping
fps = video.get(cv2.CAP_PROP_FPS)
interval = 5 # seconds
skip_frames = int(fps * interval)
skip = False

ssocr = SSOCR()
capacitance = []
units = []
frame_counter = 0
window_names = ['Cropped', 'Number', 'Unit', 'Result']

while True:
    # read a frame from the video
    ret, frame = video.read()
    frame_counter += 1

    # if the frame was not successfully read, exit the loop
    if not ret:
        break

    if skip and frame_counter < skip_frames:
        continue
    frame_counter = 0 # reset frame counter only if you want to skip for every interval
    # no need to reset frame counter if you want to skip first t seconds only
    # probably two counters are needed if you want to do both

    # crop the frame to get the region of interest
    cropped = frame[1070:1185, 540:882]
    number_region = cropped[:, 0:300]
    unit_region = cropped[82:, 300:328]

    # run the SSOCR algorithm
    number, digit_img = ssocr.run_digit(number_region)
    unit = ssocr.run_unit(unit_region)
    unit = unit + 'F'

    # append the capacitance and unit to the list
    capacitance.append(number)
    units.append(unit)

    # display the frames
    to_display = [cropped, number_region, unit_region, digit_img]
    assert len(to_display) == len(window_names) # make sure the number of images and window names are the same
    for i in range(len(to_display)):
        cv2.imshow(window_names[i], to_display[i])

    # if the 'q' key is pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

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
