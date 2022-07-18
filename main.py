from tensorflow.keras.models import model_from_json
import numpy as np
import cv2


file = open('saved_model_weights', 'r')
model_json = file.read()
file.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights('h5_file')

# classes = {0: 'Not Fencing', 1: 'Fencing'}
original_vid = cv2.VideoCapture('video144p.avi')
frame_count = int(original_vid.get(cv2.CAP_PROP_FRAME_COUNT))
dim_y, dim_x = original_vid.get(cv2.CAP_PROP_FRAME_WIDTH), original_vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
image_reduction_y = 64 / dim_y  # 64x36 standard (quarter of 256x144)
image_reduction_x = 36 / dim_x

output_vid = []
skip_frames = 10
i = 0
fencing = False

while original_vid.isOpened():
    ret, frame = original_vid.read()
    if ret:
        if i % skip_frames == 0:
            fencing = False
            print(str(i) + ' of ' + str(frame_count))
            image = cv2.resize(frame, (0, 0), None, image_reduction_y, image_reduction_x)  # resize to standard
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            image = np.array(image)
            # image = np.expand_dims(image, axis=0)
            image = image / 255
            image = image.reshape(-1, 36, 64, 1)
            if np.argmax(loaded_model.predict(image)) == 1:
                output_vid.append(frame)
                fencing = True
        elif fencing:
            output_vid.append(frame)
        else:
            fencing = False
    else:
        break
    i += 1

video = cv2.VideoWriter('OUTPUTVID.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (int(dim_y), int(dim_x)))
for frame in output_vid:
    video.write(frame)
video.release()
