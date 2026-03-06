import streamlit as st
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

class_names = ['Apple Scab Leaf', 
               'Corn leaf blight', 
               'Early Blight', 
               'Healthy', 
               'Late Blight', 
               'Leaf Spot', 
               'Rust Leaf', 
               'Squash Powdery mildew leaf', 
               'Tomato leaf mosaic virus', 
               'Tomato leaf yellow virus', 
               'Tomato mold leaf', 
               'grape leaf black rot']

@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="plant_disease_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image):
    image = image.resize((224, 224))
    img = np.array(image).astype(np.float32)

    # MobileNetV2 preprocessing
    img = (img / 127.5) - 1

    img = np.expand_dims(img, axis=0)
    return img

def predict(image):

    input_data = preprocess_image(image)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data

st.title("🌱 Plant Disease Detector")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    predictions = predict(image)

    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    st.write("Prediction:", class_names[predicted_class])
    st.write("Confidence:", float(confidence))
