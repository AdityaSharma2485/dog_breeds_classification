import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import ImageTk, Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

labels = ['Beagle', 'Bulldog', 'Doberman', 'German Shepard', 'Labrador', 'Lhasa', 'Pitbull', 'Pug', 'Rottweiler', 'Siberian Husky']
label_encoder = LabelEncoder()
label_encoder.fit(labels)
# Load the pre-trained model
model = load_model("E:\My_Projects\Multiclass_dog_breeds_10\model_dog_breeds_10_classes.h5")


# Function to perform prediction
def predict_image():
    image = Image.open(image_path.get())
    image = image.resize((256, 256))  # Resize image to match model input size
    image_array = img_to_array(image) / 255.0  # Convert image to array and normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add an extra dimension

    # Predict class probabilities
    predictions = model.predict(image_array)

    # Convert probabilities to class labels
    predicted_labels = np.argmax(predictions, axis=1)

    # Decode predicted labels using label_encoder
    decoded_labels = label_encoder.inverse_transform(predicted_labels)

    # Update prediction label
    prediction_label.config(text=f'Prediction: {decoded_labels[0]}')

    # Display the image
    image = image.resize((300, 300))  # Resize image for display
    img_tk = ImageTk.PhotoImage(image)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # Pack the footer label to ensure it stays at the bottom
    footer_label.pack(side='bottom', pady=10, fill='x', expand=True)


def browse_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    image_path.set(filepath)


# Create the main window
window = tk.Tk()
window.title("Dog vs. Cat Classification")
window.geometry("600x550")
window.configure(bg='white')

# Title label
title_label = tk.Label(window, text="Dog Breeds Classification", font=('Arial', 24, 'bold'), fg='darkblue', bg='white')
title_label.pack(pady=20)

# Image path entry
path_frame = tk.Frame(window, bg='white')
path_frame.pack(pady=10)
image_path = tk.StringVar()
image_entry = tk.Entry(path_frame, textvariable=image_path, width=40, font=('Arial', 12))
image_entry.pack(side='left', padx=5)
browse_button = tk.Button(path_frame, text="Browse", font=('Arial', 12), relief='solid', command=browse_image)
browse_button.pack(side='left')

# Image and prediction display
image_frame = tk.Frame(window, bg='white')
image_frame.pack(pady=20)
image_label = tk.Label(image_frame)
image_label.pack(side='left', padx=20)
prediction_label = tk.Label(window, text="Prediction: ", font=('Arial', 18, 'bold'), fg='darkblue', bg='white')
prediction_label.pack()

# Predict button
predict_button = tk.Button(window, text="Predict", font=('Arial', 14, 'bold'), relief='solid', bg='lightblue', fg='white', command=predict_image)
predict_button.pack(pady=30)

# Footer label
footer_label = tk.Label(window, text="© 2023 MulticlassDogBreeds Inc. All rights reserved.", font=('Arial', 10), fg='gray', bg='white')
footer_label.pack(side='bottom', pady=10, fill='x', expand=True)

# Run the main window loop
window.mainloop()


