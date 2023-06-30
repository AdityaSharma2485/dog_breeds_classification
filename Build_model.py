#!/usr/bin/env python
# coding: utf-8

# In[100]:


import os
import cv2
import numpy as np

IMG_SIZE = (256, 256)  # Set the desired image size for resizing

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, IMG_SIZE)  # Resize the image to a uniform shape
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels)

def preprocess_data(train_folder, valid_folder):
    train_images, train_labels = load_images_from_folder(train_folder)
    valid_images, valid_labels = load_images_from_folder(valid_folder)
    
    print("Loaded training data. Shape:", train_images.shape)
    print("Loaded validation data. Shape:", valid_images.shape)
    
    return train_images, train_labels, valid_images, valid_labels
# Specify the paths to your train and test folders
train_folder = r"E:\My_Projects\Datasets\Multiclass_dog_breeds_10\train"
valid_folder = r"E:\My_Projects\Datasets\Multiclass_dog_breeds_10\valid"

# Preprocess the data
train_images, train_labels, valid_images, valid_labels = preprocess_data(train_folder, valid_folder)


# In[39]:


import matplotlib.pyplot as plt

def get_label_name(label):
    return label  # Return the label as it is

# Display two images from the training data
for image, label in zip(train_images[1088:1092], train_labels[1088:1092]):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
    plt.show()



# In[40]:


import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Assuming train_labels and valid_labels are lists of class labels

# Encode the labels using LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(train_labels)

# Transform the train_labels and valid_labels into numerical values
train_labels_encoded = label_encoder.transform(train_labels)
valid_labels_encoded = label_encoder.transform(valid_labels)

# Perform one-hot encoding on the encoded labels
num_classes = len(label_encoder.classes_)
train_labels_encoded = tf.keras.utils.to_categorical(train_labels_encoded, num_classes)
valid_labels_encoded = tf.keras.utils.to_categorical(valid_labels_encoded, num_classes)


# In[41]:


print(train_images.shape)
print(train_labels_encoded.shape)


# In[58]:


import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# Load the pre-trained VGG16 model without the classification layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# Freeze the weights of the base model
base_model.trainable = False

# Create a new model
model = models.Sequential()

# Add the pre-trained base model
model.add(base_model)

# Add your own classification layers
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))


# In[59]:


model.summary()


# In[60]:


# Compile the model
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[61]:


from tensorflow.keras.callbacks import EarlyStopping

# Create an instance of EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train the model with early stopping
history = model.fit(train_images, train_labels_encoded,batch_size = 32, epochs=20,validation_data=(valid_images, valid_labels_encoded), callbacks=[early_stopping])

# Access the training history
train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']

# Print the training and validation accuracies
for epoch in range(len(train_loss)):
    print(f"Epoch {epoch+1}/{len(train_loss)}")
    print(f"Training Loss: {train_loss[epoch]}, Accuracy: {train_acc[epoch]}")
    print(f"Validation Loss: {val_loss[epoch]}, Accuracy: {val_acc[epoch]}")


# In[99]:


from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2
import numpy as np

# Assuming you have already loaded and compiled your model
# Assuming you have already defined and fitted your label_encoder

# Load and preprocess the image
image_path = r"C:\Users\adity\Downloads\rott_test.jpg"
image = load_img(image_path, target_size=(256, 256))
image_array = img_to_array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Predict class probabilities
predictions = model.predict(image_array)

# Convert probabilities to class labels
predicted_labels = np.argmax(predictions, axis=1)

print(predictions)

# Decode predicted labels using label_encoder
decoded_labels = label_encoder.inverse_transform(predicted_labels)

# Print the predicted class label
print(decoded_labels[0])

# Display the image
plt.imshow(image)
plt.axis('off')
plt.show()


# In[90]:


model.save('model_dog_breeds_10_classes.h5')

