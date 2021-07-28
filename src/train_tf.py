""" Training script in Tensorflow """

import tensorflow as tf
from tensorflow.keras import layers

img_shape = (150, 150)

DataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False, samplewise_center=False,
    featurewise_std_normalization=False, samplewise_std_normalization=False,
    zca_whitening=False, zca_epsilon=1e-06, rotation_range=10, width_shift_range=0.1,
    height_shift_range=0.1, brightness_range=None, shear_range=0.2, zoom_range=0.3,
    channel_shift_range=0.0, fill_mode='nearest', cval=0.0,
    horizontal_flip=True, vertical_flip=True, rescale=None,
    preprocessing_function=None, validation_split=0.3, data_format=None, dtype=None
)

train_data = DataGenerator.flow_from_directory(
    "intel_images/seg_train",
    target_size= img_shape,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=5,
    shuffle=True,
    subset="training"
)

val_data = DataGenerator.flow_from_directory(
    "intel_images/seg_train",
    target_size= img_shape,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=5,
    shuffle=True,
    subset="validation"
)

test_data = DataGenerator.flow_from_directory(
    "intel_images/seg_test",
    target_size=img_shape,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=5,
    shuffle=True
)


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

model.compile(optimizer = 'adam', loss = tf.keras.losses.CategoricalCrossentropy()  , metrics=['accuracy'])

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

try:
    model.load_weights("tmp")
except Exception as e:
    print(e)

history = model.fit(train_data,
                    batch_size=128,
                    epochs=100,
                    validation_data=val_data,
                    callbacks=[modelcheckpoint, early_stopping, csv_logger, remote_monitor])

model.save("saved_model")
res = model.evaluate(test_data)

print(res)

