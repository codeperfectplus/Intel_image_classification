""" Training scrip(t in Tensorflow """
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import preprocess_input

root_dir = os.path.dirname(os.path.abspath(os.path.abspath(__name__)))
train_dir = os.path.join(root_dir, "intel_images/seg_train")
test_dir = os.path.join(root_dir, "intel_images/seg_test")
img_size = (150, 150)
img_shape = (150, 150, 3)
batch_size = 128
num_classes = len(os.listdir(train_dir))
idx_to_name = os.listdir(train_dir)
name_to_idx = dict([(v, k) for k, v in enumerate(idx_to_name)])

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

train_df, val_df = data_to_df(train_dir, subset="train")

class CustomDataGenerator:
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
    
    def __len__(self):
        return len(self.train_len/self.batch_size)

    def preprocessing(self, filename, label):
        """ converting filenames into numpy array and applying image preprocessing """
        img = tf.keras.preprocessing.image.load_img(filename)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.resize(img, self.img_shape)
        img = preprocess_input(img)
        img = tf.keras.preprocessing.image.random_shift(img, 0.2, 0.3)
        return img, label

    def generate(self):
        ''' Generator function to yield img and label '''
        num_samples = self.data_frame.shape[0]
        while True:
            if self.shuffle_data:
                self.data_frame = shuffle(self.data_frame)
            filenames = self.data_frame["filenames"].tolist()
            labels = self.data_frame["labels"].tolist()

            for offset in range(0, num_samples, self.batch_size):
                filenames = filenames[offset:offset+self.batch_size]
                labels = labels[offset:offset+self.batch_size]
                X_train = []
                y_train = []

                for filename, label in zip(filenames, labels):

                    img, label = self.preprocessing(filename, label)

                    X_train.append(img)
                    y_train.append(label)

                X_train = np.array(X_train)
                y_train = np.array(y_train)

                yield X_train, y_train

# creating train and validation data
train_data = CustomDataGenerator(train_df, 
                                 batch_size=batch_size, 
                                 img_shape=img_shape).generate()
val_data = CustomDataGenerator(val_df, 
                               batch_size=10, img_shape=img_shape).generate()


class buildModel(tf.keras.Model):
    def __init__(self):
        super(buildModel, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3))
        self.max_pool = layers.MaxPooling2D(2,2)
        self.conv2 = layers.Conv2D(32, (3, 3), activation = 'relu')
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation=tf.nn.relu)
        self.dense2 = layers.Dense(6, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

model = buildModel()

model.compile(optimizer = 'adam', 
              loss = tf.keras.losses.CategoricalCrossentropy(), 
              metrics=['accuracy'])

modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
    "tmp",
    monitor="val_loss",
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
    options=None)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=5,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)

remote_monitor = tf.keras.callbacks.RemoteMonitor(
    root="http://localhost:5000",
    path="/publish/epoch/end/",
    field="data",
    headers=None,
    send_as_json=False,
)

csv_logger = tf.keras.callbacks.CSVLogger("log.csv", separator=",", append=False)

callbacks = [modelcheckpoint, early_stopping, csv_logger, remote_monitor]

try:
    model.load_weights("tmp")
except Exception as e:
    print(e)

x, y = next(train_data)

history = model.fit(train_data,
                    batch_size=batch_size,
                    epochs=100,
                    steps_per_epoch = len(train_samples)//batch_size,
                    validation_data=val_data,
                    callbacks= callbacks)

test_gen = CustomDataGenerator(test_dir)
test_samples = test_gen.load_samples()
test_data = test_gen.generate(test_samples, batch_size=128)

model.save("saved_model")
res = model.evaluate(test_data)

print(res)