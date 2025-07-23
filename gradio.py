import numpy as np
from PIL import Image
import requests, os
from io import BytesIO
import gradio as gr
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress messy terminal messages
import tensorflow as tf

model = tf.keras.models.load_model('best_model.keras') # Load the model
class_names = ['Drowsy', 'Non drowsy']

# Sourcing sample images from web
url_1 = requests.get("https://source.roboflow.com/CXVu8FaWCDf9iUjw4NOfU9gfgUG3/00w6zrKrEaWmwN204HM7/original.jpg")
img_1 = Image.open(BytesIO(url_1.content))

url_2 = requests.get("https://source.roboflow.com/Fa12m5ZY57XUlyVaUuz5DqMjXoo1/01FON0wNXjwwhtzRs51K/original.jpg")
img_2 = Image.open(BytesIO(url_2.content))

url_3 = requests.get("https://source.roboflow.com/Y756mSC6kpZqdTzOg6u299tKXxg1/1HthKldvKlvobu6tTdPF/original.jpg")
img_3 = Image.open(BytesIO(url_3.content))

def classify_image(Image):
    img = np.resize(Image, (224, 224, 3))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    raw_pred = model.predict(img_array)
    pred = (raw_pred >= 0.5).astype(int).item()
    return f"Raw score: {np.round(raw_pred.item(), 4)}\nClass: {class_names[pred]}"

gr.Interface(fn = classify_image,
             inputs=gr.Image(width=224, height=224),
             outputs=gr.Textbox(label="Answer"),
             title="Driver Drowsiness Detection",
             examples = [img_1, img_2, img_3]).launch(debug=True)
