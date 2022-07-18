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


# video_name = 'video144p.avi'
# sgd = SGD(learning_rate=0.01)
# context = 1
# frame_rate = 10
# epochs = 100
# batch_size = 8
# image_reduction = 0.25
# train_size_reduce = 1
# train_test_ratio = 0.8
# classes = {1: 'Fencing', 0: 'Not Fencing'}
# image_shape = video_split(video_name, frame_rate, image_reduction)    # downloads every frame_rate image and returns the shape of the entire dataset
# train_images, length = split_images(context, image_shape.shape)    # train_images are groups of 9 images from training vid
# get_mode = int(input('Create new data (0), use previous data (1), continue classifying previous data (2): '))
# train_labels, label_count, number_of_classes = previous_data(get_mode)
#
# if train_labels == None:
#     train_labels, label_count, number_of_classes = image_classifier(context)    # list (will become numpy array) of lists with 2 integers inside; has shape (len(), 2) and # of labels

# number_of_predictions = input('How many images would you like to predict on? ("n" for none) ')

# randomly shuffle train images and labels simultaneously by index
images_data = []
labels_data = []
with open('labels.txt') as f:
    labels = f.readline()
    for i in labels:
        labels_data.append(int(i))
        labels_data.append(int(i))
    print('Classified so far - Fencing:', labels.count('1'), 'Not fencing:', labels.count('0'))
    label_count = len(labels)

for i in range(label_count):
    path = 'classified_frames/' + str(i + 1) + '.jpg'
    frame = cv2.imread(path, 0)
    image = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
    images_data.append(image)
    images_data.append(cv2.flip(image, 1))

print(np.shape(np.array(images_data)))
print(np.shape(np.array(labels_data)))
images_data = np.array(images_data)
labels_data = np.array(labels_data)
images_data = images_data / 255.0
images_data = images_data.reshape(-1, 72, 128, 1)
labels_data = to_categorical(labels_data)

permutation = np.random.permutation(len(labels_data))

images_data = images_data[permutation]
labels_data = labels_data[permutation]
train_images = images_data[:int(len(images_data)//1.25)]
train_labels = labels_data[:int(len(labels_data)//1.25)]
test_images = images_data[int(len(images_data)//1.25):]
test_labels = labels_data[int(len(labels_data)//1.25):]


model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=(72, 128, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
# model.add(layers.Dropout(0.2))
# model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
# model.add(layers.Dense(2))


model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),    # process the binary label inputs
              # optimizer=SGD(learning_rate=0.01),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=30, batch_size=128,
                    validation_data=(test_images, test_labels))

# show training vs. validation outcome to test for overfitting, underfitting, or model bugs
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(model.summary())
print('Test accuracy:', test_acc)

####################
open('saved_model_weights', 'w').close()
json_file = model.to_json()
with open('saved_model_weights', "w") as file:
    file.write(json_file)

model.save_weights('h5_file')
print('Model saved')
quit()

########################################################################################################################
permutation = np.random.permutation(len(train_labels))

train_images = train_images[:label_count]    # reduce size of training data to match number of labels classified
train_images = train_images[permutation]
train_images = train_images / 255    # turn integers into scalars for easier processing
if number_of_predictions != 'n':
    predict_image = train_images[len(train_images) - int(number_of_predictions):]
train_images = tf.cast(train_images, dtype='float32')
train_images = train_images[:int(train_images.shape[0] * train_size_reduce)]    # reduce the size of training data
test_images = train_images[int(train_images.shape[0] * train_test_ratio):]
train_images = train_images[:int(train_images.shape[0] * train_test_ratio)]    # split data into training and testing

train_labels = np.array(train_labels)    # turn labels into numpy array
train_labels = train_labels[permutation]
train_labels = tf.cast(train_labels, dtype='float32')
train_labels = train_labels[:int(train_labels.shape[0] * train_size_reduce)]    # reduce the label size to match training data size
test_labels = train_labels[int(train_labels.shape[0] * train_test_ratio):]
train_labels = train_labels[:int(train_labels.shape[0] * train_test_ratio)]    # split labels into training and testing
train_labels = to_categorical(train_labels)    # turn labels into one-hot vectors (binary)
test_labels = to_categorical(test_labels)

if input('Preview first image of random 9 sets? y/n: ').lower().startswith('y'):    # display 9 images with labels for quality check
    for i in range(9):
        random_index = random.randint(0, len(train_labels) - 1)
        plt.subplot(3, 3, i + 1)
        plt.imshow(train_images[random_index][0])
        plt.title(classes[np.argmax(train_labels[random_index])])
    plt.show()

print('Train images shape:', train_images.shape)
print('Train labels shape:', train_labels.shape)
print('Test images shape:', test_images.shape)
print('test labels shape:', test_labels.shape)

model = models.Sequential()
model.add(layers.Conv3D(64, (1, 3, 3), padding='same', activation='relu', input_shape=(train_images.shape[1:])))
model.add(layers.MaxPooling3D((1, 2, 2)))
model.add(layers.Conv3D(128, (1, 3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling3D((1, 2, 2)))
model.add(layers.Dropout(0.3))
model.add(layers.Conv3D(256, (1, 3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling3D((1, 2, 2)))
model.add(layers.Dropout(0.4))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid'))    # maybe something is up with this last layer causing model not to learn

print('min and max scalers for first image', np.min(train_images[0][0]), np.max(train_images[0][0]))
print('Training size:', len(train_labels))
print('Classes:', number_of_classes)

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),    # process the binary label inputs
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, shuffle=True,
                    validation_data=(test_images, test_labels))

# show training vs. validation outcome to test for overfitting, underfitting, or model bugs
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

print('accuracy: ', acc)
print('val acc: ', val_acc)
print('loss: ', loss)
print('val_loss: ', val_loss)

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# Predict labels for each video image
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('Test accuracy:', test_acc)

plt.show()

if number_of_predictions != 'n':
    number_of_predictions = int(number_of_predictions)
    predict_image = tf.cast(predict_image, dtype='float32')
    print(predict_image.shape)
    model_prediction = model.predict(predict_image)

    plt.figure(figsize=(8, 8))
    for i in range(predict_image.shape[0]):
        if number_of_predictions == 1:
            x, y = 1, 1
        elif number_of_predictions == 3:
            x, y = 1, 3
        elif 5 <= number_of_predictions < 10:
            x, y = 3, 3
        elif 9 < number_of_predictions <= 16:
            x, y = 4, 4
        else:
            break

        if np.argmax(model_prediction[i]) == 0:
            index = 0
        else:
            index = 1

        plt.subplot(x, y, i + 1)
        plt.imshow(predict_image[i][math.ceil(predict_image.shape[1] / 2) - 1])
        plt.title(
            classes[np.argmax(model_prediction[i])] + ' (with ' + str(round_up(model_prediction[i][index] * 100, 2))
            + '% confidence)', fontsize=6)

    plt.show()


open('saved_model_weights', 'w').close()
json_file = model.to_json()
with open('saved_model_weights', "w") as file:
    file.write(json_file)

model.save_weights('h5_file')
print('Model saved')
