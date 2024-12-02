# -*- coding: utf-8 -*-

import os
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from keras import layers, Model
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from skimage import measure

from google.colab import drive
drive.mount('/content/drive')

def load_dataset(image_dir, mask_dir, img_size=(512, 512)):
    images = []
    masks = []

    image_filenames = sorted(os.listdir(image_dir))
    mask_filenames = sorted(os.listdir(mask_dir))

    for img_file, mask_file in zip(image_filenames, mask_filenames):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        image = load_img(img_path, target_size=img_size)
        mask = load_img(mask_path, target_size=img_size, color_mode="grayscale")

        image = img_to_array(image) / 255.0
        mask = img_to_array(mask) / 255.0
        mask = np.round(mask)

        images.append(image)
        masks.append(mask)


    return np.array(images), np.array(masks)

image_dir = "/content/drive/MyDrive/AIML for Global Health/Images_left_eye_corrected"
mask_dir = "/content/drive/MyDrive/AIML for Global Health/SegmentationClass"
X, Y = load_dataset(image_dir, mask_dir)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
print("train data shape:", X_train.shape)
print("validation data shape:", X_val.shape)


image_dir1 = "/content/drive/MyDrive/AIML for Global Health/Images_left_palm_corrected"
mask_dir1 = "/content/drive/MyDrive/AIML for Global Health/SegmentationClass"
X1, Y1 = load_dataset(image_dir1, mask_dir1)
X1_train, X1_val, Y1_train, Y1_val = train_test_split(X1, Y1, test_size=0.2, random_state=42)
print("train data shape:", X1_train.shape)
print("validation data shape:", X1_val.shape)

image_dir2 = "/content/drive/MyDrive/AIML for Global Health/Images_right_fingernail_corrected"
mask_dir2 = "/content/drive/MyDrive/AIML for Global Health/SegmentationClass"
X2, Y2 = load_dataset(image_dir2, mask_dir2)
X2_train, X2_val, Y2_train, Y2_val = train_test_split(X2, Y2, test_size=0.2, random_state=42)
print("train data shape:", X2_train.shape)
print("validation data shape:", X2_val.shape)


def unet_model(input_size=(512, 512, 3)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    # Decoder
    u1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = Model(inputs, outputs)
    return model

model_eyelid = unet_model()
model_eyelid.summary()
model_eyelid.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model_eyelid.fit(
    X_train, Y_train,
    epochs=10,
    batch_size=8,  
    validation_data=(X_val, Y_val)
)
model_eyelid.save('/content/drive/MyDrive/AIML for Global Health/eyelid_model.h5')
model_eyelid.save_weights('/content/drive/MyDrive/AIML for Global Health/eyelid_model.weights.h5')
with open('/content/drive/MyDrive/AIML for Global Health/eyelid_model.pkl', 'wb') as file:
    pickle.dump(model_eyelid, file)


model_palm = unet_model()
model_palm.summary()
model_palm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_1 = model_palm.fit(
    X1_train, Y1_train,
    epochs=20,
    batch_size=8,  
    validation_data=(X1_val, Y1_val)
)
model_palm.save('/content/drive/MyDrive/AIML for Global Health/palm_model.h5')
model_palm.save_weights('/content/drive/MyDrive/AIML for Global Health/palm_model.weights.h5')
with open('/content/drive/MyDrive/AIML for Global Health/palm_model.pkl', 'wb') as file:
    pickle.dump(model_palm, file)


model_fingernail = unet_model()
model_fingernail.summary()
model_fingernail.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=2,         
    verbose=1,           
    restore_best_weights=True  
)

history_2 = model_fingernail.fit(
    X2_train, Y2_train,
    epochs=15,
    batch_size=8,  # 根据内存大小调整
    validation_data=(X2_val, Y2_val)
)
model_fingernail.save('/content/drive/MyDrive/AIML for Global Health/fingernail_model.h5')
model_fingernail.save_weights('/content/drive/MyDrive/AIML for Global Health/fingernail_model.weights.h5')
with open('/content/drive/MyDrive/AIML for Global Health/fingernail_model.pkl', 'wb') as file:
    pickle.dump(model_fingernail, file)





def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0 
    image = np.expand_dims(image, axis=0) 
    return image


def predict_segmentation(model, preprocessed_image):
    prediction = model.predict(preprocessed_image)
    prediction = np.squeeze(prediction, axis=0) 
    prediction = (prediction > 0.5).astype(np.uint8)  
    return prediction


def visualize_segmentation(original_image_path, prediction):
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.subplot(1, 2, 2)
    plt.title("Segmentation")
    plt.imshow(prediction, cmap='gray')
    plt.show()


def save_prediction_as_png(original_image_path, prediction, output_dir):
    image_filename = os.path.basename(original_image_path)
    image_name, _ = os.path.splitext(image_filename)

    output_filename = f"{image_name}.png"
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    # cv2.imwrite(output_path, prediction * 255) 

    image = prediction * 255
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    labels = measure.label(binary, connectivity=2)
    properties = measure.regionprops(labels)
    if len(properties) == 0:
        print("No regions found")
        largest_region_mask = np.zeros_like(binary)
    else:
        largest_region = max(properties, key=lambda x: x.area)
        largest_region_mask = np.zeros_like(binary)
        largest_region_mask[labels == largest_region.label] = 255

    cv2.imwrite(output_path, largest_region_mask)

    print(f"Prediction saved to {output_path}")


def process_single_image(image_path, model, target_size, output_dir):
    preprocessed_image = preprocess_image(image_path, target_size)
    prediction = predict_segmentation(model, preprocessed_image)
    save_prediction_as_png(image_path, prediction, output_dir)



input_dir = '/content/drive/MyDrive/AIML for Global Health/Images_right_eye_corrected'  
output_dir = '/content/drive/MyDrive/AIML for Global Health/SegmentationClass'  
target_size = (512, 512) 
model = model_eyelid

for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg')):
        image_path = os.path.join(input_dir, filename)
        process_single_image(image_path, model, target_size, output_dir)