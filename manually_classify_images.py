import cv2
import os
import math


for file in os.listdir('classify_ith_frame_visual/'):  # remove old files
    left_over_path = 'classify_ith_frame_reduction' + file
    if os.path.isfile(left_over_path):
        os.unlink(os.path.join('classify_ith_frame_reduction/', file))
    os.unlink(os.path.join('classify_ith_frame_visual/', file))


with open('labels.txt') as f:
    label = f.readline()
    print('Classified so far - Fencing:', label.count('1'), 'Not fencing:', label.count('0'))
    buffer = len(label) + 1
i = 0
skip_frames = 100
labels = ''
# vid = 'SantarelliBorel.mp4'
vid = 'video144p.avi'
cap = cv2.VideoCapture(vid)
# dim_y, dim_x = vid  # get somehow
dim_y, dim_x = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(dim_y, dim_x)
image_reduction_y = 64 / dim_y  # 64x36 standard (quarter of 256x144)
image_reduction_x = 36 / dim_x


while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if i % skip_frames == 0:
            print(buffer, i, skip_frames, str(buffer + i // skip_frames))
            cv2.imwrite('classify_ith_frame_visual/' + str(buffer + i // skip_frames) + '.jpg', frame)
            image = cv2.resize(frame, (0, 0), None, image_reduction_y, image_reduction_x)  # resize to standard
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            cv2.imwrite('classify_ith_frame_reduction/' + str(buffer + i // skip_frames) + '.jpg', image)  # store frame
    else:
        break
    i += 1


for frame_number in range(buffer, i // skip_frames + buffer + 1):
    print(frame_number, end=" - ")
    input_ = input('0: Not fencing, 1: Fencing, 2: quit and save - ')
    if input_ not in '01':
        break
    labels += input_
    os.rename('classify_ith_frame_reduction/' + str(frame_number) + '.jpg', 'classified_frames/' + str(frame_number) + '.jpg')
    os.unlink('classify_ith_frame_visual/' + str(frame_number) + '.jpg')

with open('labels.txt', 'a') as f:
    f.write(labels)
    f.close()
