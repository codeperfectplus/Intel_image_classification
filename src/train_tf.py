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
    
    def __init__(self, dataframe):
        self.dataframe = dataframe 
    
    def load_samples(self):
        data = self.dataframe[["filenames", "labels"]]
        filenames = list(data["filenames"])
        labels = list(data["labels"])
        
        samples = []
        for name, label in zip(filenames, labels):
            samples.append([name, label])
        return samples
        
    def preprocessing(self, img, label):
        img = np.resize(img, img_shape)
        img = preprocess_input(img)
        label = tf.keras.utils.to_categorical(label, num_classes)
        
        return img, label

    def generate(self, samples, batch_size=10, is_preprocessing=True):
        while True:
            samples = shuffle(samples)
            
            for offset in range(0, len(samples), batch_size):
                batch_samples = samples[offset:offset+batch_size]
                
                X_train = []
                y_train = []
                    
                for batch_sample in batch_samples:
                    filename = batch_sample[0]
                    label = batch_sample[1]
                    
                    img = np.asarray(Image.open(filename))
                    if is_preprocessing:
                        img, label = self.preprocessing(img, label)
                    
                    X_train.append(img)
                    y_train.append(label)
                
                X_train, y_train = np.array(X_train), np.array(y_train)
                
                yield X_train, y_train

train_gen = CustomDataGenerator(train_df)
val_gen = CustomDataGenerator(val_df)

train_samples = train_gen.load_samples()
val_samples = val_gen.load_samples()

train_data = train_gen.generate(train_samples)
val_data = val_gen.generate(val_samples)


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