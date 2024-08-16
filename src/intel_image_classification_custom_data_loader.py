import os
import logging
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics, layers
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration parameters
train_dir = "../input/intel-image-classification/seg_train/seg_train"
test_dir = "../input/intel-image-classification/seg_test/seg_test"
epochs = 50
img_size = (150, 150)
img_shape = (*img_size, 3)
batch_size = 256
num_classes = len(os.listdir(train_dir))
idx_to_name = os.listdir(train_dir)
name_to_idx = {v: k for k, v in enumerate(idx_to_name)}

def data_to_df(data_dir, subset=None):
    """
    Convert the dataset directory into a pandas DataFrame.
    
    Args:
        data_dir (str): Path to the dataset directory.
        subset (str, optional): If 'train', splits data into training and validation sets. Defaults to None.
    
    Returns:
        pd.DataFrame: DataFrame containing image file paths and corresponding labels.
        If subset is 'train', returns two DataFrames (train_df, val_df).
    """
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
        train_df, val_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=10)
        return train_df, val_df
    
    return df

logging.info("Converting data directory to dataframe")
train_df, val_df = data_to_df(train_dir, subset="train")

class CustomDataGenerator(tf.keras.utils.Sequence):
    """
    Custom DataGenerator to load images and their corresponding labels.
    
    Args:
        data_frame (pd.DataFrame): DataFrame containing file paths and labels.
        batch_size (int, optional): Number of samples per batch. Defaults to 10.
        img_shape (tuple, optional): Shape of the images (height, width, channels). Defaults to None.
        augmentation (bool, optional): Whether to apply data augmentation. Defaults to True.
        num_classes (int, optional): Number of classes. Defaults to None.
    
    Returns:
        A generator that yields batches of image data and labels.
    """
    
    def __init__(self, data_frame, batch_size=10, img_shape=None, augmentation=True, num_classes=None):
        self.data_frame = data_frame
        self.train_len = self.data_frame.shape[0]
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.augmentation = augmentation
        logging.info(f"Found {self.data_frame.shape[0]} images belonging to {self.num_classes} classes")

    def __len__(self):
        """
        Returns the number of batches per epoch.
        
        Returns:
            int: Number of batches per epoch.
        """
        return int(np.ceil(self.train_len / self.batch_size))

    def on_epoch_end(self):
        """
        Shuffle data at the end of each epoch.
        """
        self.data_frame = shuffle(self.data_frame)

    def __data_augmentation(self, img):
        """
        Apply data augmentation to an image.
        
        Args:
            img (np.ndarray): Input image.
        
        Returns:
            np.ndarray: Augmented image.
        """
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img
        
    def __get_image(self, file_id):
        """
        Load and preprocess an image from the file path.
        
        Args:
            file_id (str): File path of the image.
        
        Returns:
            np.ndarray: Preprocessed image.
        """
        try:
            img = Image.open(file_id)
            img = img.resize(self.img_shape[:2])
            img = np.asarray(img)
            img = preprocess_input(img)
            if self.augmentation:
                img = self.__data_augmentation(img)
            return img
        except (UnidentifiedImageError, OSError) as e:
            logging.error(f"Error loading image {file_id}: {e}")
            return np.zeros(self.img_shape)

    def __get_label(self, label_id):
        """
        Get the label for a given image.
        
        Args:
            label_id (int): Label ID.
        
        Returns:
            int: Label ID (as is).
        """
        return label_id

    def __getitem__(self, idx):
        """
        Generate one batch of data.
        
        Args:
            idx (int): Index of the batch.
        
        Returns:
            tuple: Batch of images and labels.
        """
        batch_x = self.data_frame["filenames"][idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.data_frame["labels"][idx * self.batch_size:(idx + 1) * self.batch_size]
        x = [self.__get_image(file_id) for file_id in batch_x] 
        y = [self.__get_label(label_id) for label_id in batch_y]

        return np.array(x), np.array(y)

logging.info("Creating train and validation data generators")
train_data = CustomDataGenerator(train_df, batch_size=batch_size, img_shape=img_shape, num_classes=num_classes)
val_data = CustomDataGenerator(val_df, batch_size=batch_size, img_shape=img_shape, num_classes=num_classes)

def build_model(img_shape, num_classes):
    """
    Build a VGG16-based convolutional neural network model.
    
    Args:
        img_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.
    
    Returns:
        tf.keras.Model: Compiled Keras model.
    """
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
    outputs = layers.Dense(num_classes, activation="softmax")(out)

    model = tf.keras.Model(inputs, outputs)
    return model

model = build_model(img_shape, num_classes)
model.compile(optimizer=optimizers.Adam(learning_rate=0.001), 
              loss=losses.SparseCategoricalCrossentropy(), 
              metrics=[metrics.SparseCategoricalAccuracy()])

loss_fn = losses.SparseCategoricalCrossentropy()
train_acc_metrics = metrics.SparseCategoricalAccuracy()
val_acc_metrics = metrics.SparseCategoricalAccuracy()

# Model checkpoint callback to save the best model based on validation accuracy
checkpoint_path = "best_model.h5"
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                       monitor='val_sparse_categorical_accuracy',
                                       save_best_only=True,
                                       save_weights_only=True,
                                       verbose=1)

@tf.function
def train_step(x, y):
    """
    Perform a single training step.
    
    Args:
        x (tf.Tensor): Batch of input images.
        y (tf.Tensor): Batch of labels.
    
    Returns:
        tf.Tensor: Loss value for the current batch.
    """
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metrics.update_state(y, logits)
    return loss_value

@tf.function
def test_step(x, y):
    """
    Perform a single validation step.
    
    Args:
        x (tf.Tensor): Batch of input images.
        y (tf.Tensor): Batch of labels.
    """
    val_logits = model(x, training=False)
    val_acc_metrics.update_state(y, val_logits)

logging.info(f"Starting training for {epochs} epochs")
import time
best_val_acc = 0.0
for epoch in range(epochs):
    logging.info(f"Epoch :{epoch+1}/{epochs}")
    start_time = time.perf_counter()
    
    # Training
    for step, (x_batch_train, y_batch_train) in tqdm(enumerate(train_data), total=len(train_data)):
        loss_value = train_step(x_batch_train, y_batch_train)
        
        if (step % 50) == 0: 
            logging.info(f"Step: {step} Training loss :{loss_value}")
    
    train_acc = train_acc_metrics.result()
    logging.info(f"Training Accuracy: {train_acc:.4f}")
    train_acc_metrics.reset_states()

    # Validation
    for x_batch_val, y_batch_val in val_data:
        test_step(x_batch_val, y_batch_val)
    
    val_acc = val_acc_metrics.result()
    logging.info(f"Validation Accuracy: {val_acc:.4f}")
    val_acc_metrics.reset_states()

    # Checkpoint saving based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_weights(checkpoint_path)
        logging.info(f"Model checkpoint saved at epoch {epoch + 1} with validation accuracy: {val_acc:.4f}")

    end_time = time.perf_counter()
    logging.info(f"Time taken for epoch {epoch + 1}: {end_time - start_time:.2f} seconds")
