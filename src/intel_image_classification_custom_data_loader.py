# -*- coding: utf-8 -*-
"""intel-image-classification-custom-data-loader.ipynb
# Intel Image Classification

## Problem Statement
    classify the image into 6 different clases
## About the Data
    Content This Data contains around 25k images of size 150x150 distributed under 6 categories. {'buildings' -> 0, 'forest' -> 1, 'glacier' -> 2, 'mountain' -> 3, 'sea' -> 4, 'street' -> 5 }
"""

""" Training scrip(t in Tensorflow """
import os
import logging
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

train_dir = "../input/intel-image-classification/seg_train/seg_train"
test_dir = "../input/intel-image-classification/seg_test/seg_test"
img_size = (150, 150)
img_shape = (150, 150, 3)
batch_size = 256
num_classes = len(os.listdir(train_dir))
idx_to_name = os.listdir(train_dir)
name_to_idx = dict([(v, k) for k, v in enumerate(idx_to_name)])

name_to_idx

def data_to_df(data_dir, subset=None):
    df = pd.DataFrame()
    filenames = []
    labels = []
    
    for dataset in os.listdir(data_dir):
        img_list = os.listdir(os.path.join(data_dir, dataset))
        label = name_to_idx[dataset]
        
        for image in img_list:
            filenames.append(os.path.join(data_dir, dataset, image))
            labels.append(label)
        
    df["filenames"] = filenames
    df["labels"] = labels
    
    if subset == "train":
        train_df, val_df = train_test_split(df, train_size=0.8, shuffle=True,
                                            random_state=10)
        return train_df, val_df
    
    return df

print("Converting data directory to dataframe")
train_df, val_df = data_to_df(train_dir, subset="train")

class CustomDataGenerator(tf.keras.utils.Sequence):

    ''' Custom DataGenerator to load img 
    
    Arguments:
        data_frame = pandas data frame in filenames and labels format
        batch_size = divide data in batches
        shuffle = shuffle data before loading
        img_shape = image shape in (h, w, d) format
        augmentation = data augmentation to make model rebust to overfitting
    
    Output:
        Img: numpy array of image
        label : output label for image
    '''

    def __init__(self, data_frame, batch_size=10, shuffle_data=True, img_shape=None, augmentation=True):
        self.data_frame = data_frame
        self.train_len = self.data_frame.shape[0]
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data
        self.img_shape = img_shape
        self.x = self.data_frame["filenames"]
        self.y = self.data_frame["labels"]
    
    def __len__(self):
        return int(self.train_len/self.batch_size)

    def on_epoch_end(self):
        pass
    
    def __get_image(self, filename):
        img = tf.keras.preprocessing.image.load_img(filename)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.resize(img, self.img_shape)
        img = preprocess_input(img)
        img = tf.keras.preprocessing.image.random_shift(img, 0.2, 0.3)

        return np.array(img)

    def __get_label(self, filename):
        #label = tf.keras.utils.to_categorical(filename, num_classes)
        label = filename
        return label

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # read your data here using the batch lists, batch_x and batch_y
        x = [self.__get_image(filename) for filename in batch_x] 
        y = [self.__get_label(filename) for filename in batch_y]

        return np.array(x), np.array(y)

print("creating train and validation data")
train_data = CustomDataGenerator(train_df, 
                                 batch_size=batch_size, 
                                 img_shape=img_shape)
val_data = CustomDataGenerator(val_df, 
                               batch_size=10, img_shape=img_shape)

for i, (x, y) in enumerate(train_data):
    print(y)

base_model = VGG16(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
base_model.trainable= True

inputs = layers.Input(shape=img_shape)
out = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
out = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(out)
out = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(out)

out = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(out)
out = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(out)
out = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(out)

out = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(out)
out = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(out)
out = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(out)
out = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(out)

out = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(out)
out = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(out)
out = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(out)
out = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(out)

out = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(out)
out = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(out)
out = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(out)
out = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(out)

out = layers.GlobalAveragePooling2D()(out)
out = layers.Dense(128, activation="relu")(out)
out = layers.Dropout(0.5)(out)
outputs = layers.Dense(6, activation="softmax")(out)

model = tf.keras.Model(inputs, outputs, name="VGG16")

model.summary()

optimizer = optimizers.Adam()
loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
train_acc_metrics = metrics.SparseCategoricalAccuracy()
val_acc_metrics = metrics.SparseCategoricalAccuracy()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metrics.update_state(y, logits)
    return loss_value

@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metrics.update_state(y, val_logits)

epochs = 2
import time
for epoch in range(epochs):
    print(f"Epoch :{epoch}/{epochs}")
    start_time = time.perf_counter()
    for step, (x_batch_train, y_batch_train) in enumerate(train_data):
        loss_value = train_step(x_batch_train, y_batch_train)
        
        if (step % 50) == 0: 
            print(f"Step: {step} Training loss :{loss_value}")
            print(f"Seen so far: {(step + 1) * batch_size} samples") 
    
    train_acc = train_acc_metrics.result()
    print("Training Accuracy:", float(train_acc))
    train_acc_metrics.reset_states()

    for x_batch_val, y_batch_val in val_data:
        test_step(x_batch_val, y_batch_val)
    
    val_acc = val_acc_metrics.result()
    print("Validation Accuracy: ", float(val_acc))
    val_acc_metrics.reset_states()

    #print("Time Taken", time.perf_counter() - start_time)
    print("*"*80)

test_df = data_to_df(test_dir)
test_data = CustomDataGenerator(test_df, 
                                 batch_size=batch_size, 
                                 img_shape=img_shape).generate()

model.save("saved_model")
res = model.evaluate(test_data)

print(res)

# I haven't trained the model completely in this notebook