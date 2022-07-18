import tensorflow as tf
import numpy as np
import random
import math
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import cv2



def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


images_data = []
labels_data = []
with open('labels.txt') as f:  # get labels
    labels = f.readline()
    for i in labels:
        labels_data.append(int(i))
        labels_data.append(int(i))
    print('Classified so far - Fencing:', labels.count('1'), 'Not fencing:', labels.count('0'))
    label_count = len(labels)

for i in range(label_count):
    path = 'classified_frames/' + str(i + 1) + '.jpg'
    images_data.append(cv2.imread(path, 0))  # append frame of data
    images_data.append(cv2.flip(cv2.imread(path, 0), 1))  # append horizontally flipped frame

print(np.shape(np.array(images_data)))
print(np.shape(np.array(labels_data)))
images_data = np.array(images_data)
labels_data = np.array(labels_data)
images_data = images_data / 255.0  # normalize pixel values
images_data = images_data.reshape(-1, 36, 64, 1)  # reshape into 4 dimensions for model feeding
labels_data = to_categorical(labels_data)  # normalize to one-hot vectors (binary encoding)

permutation = np.random.permutation(len(labels_data))  # set permutation for random shuffle

images_data = images_data[permutation]
labels_data = labels_data[permutation]
# split 80-20 train-test
train_images = images_data[:int(len(images_data)//1.25)]
train_labels = labels_data[:int(len(labels_data)//1.25)]
test_images = images_data[int(len(images_data)//1.25):]
test_labels = labels_data[int(len(labels_data)//1.25):]


model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(36, 64, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
# model.add(layers.Dropout(0.2))
# model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
# model.add(layers.Dense(2))


model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),    # process the binary label inputs
              # optimizer=SGD(learning_rate=0.01),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=30, batch_size=32,
                    validation_data=(test_images, test_labels))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(model.summary())
print('Test accuracy:', test_acc)

####################
# save model and trained weights
open('saved_model_weights', 'w').close()
json_file = model.to_json()
with open('saved_model_weights', "w") as file:
    file.write(json_file)

model.save_weights('h5_file')
print('Model saved')
