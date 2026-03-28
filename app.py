import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

class_names = [
  'Apple Scab', 
  'Corn Leaf blight', 
  'Early Blight', 
  'Healthy', 
  'Late Blight', 
  'Leaf Spot', 
  'Rust', 
  'Squash Powdery Mildew', 
  'Tomato Leaf Mosaic', 
  'Tomato Yellow Virus', 
  'Tomato Mold', 
  'Grape Black Rot']

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="plant_disease_model.tflite")
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

    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        predictions = predict(image)

    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    st.success(f"Prediction: {class_names[predicted_class]}")
    st.info(f"Confidence: {confidence * 100:.2f}%")

    # Top 3
    st.subheader("Top Predictions:")
    top_3 = np.argsort(predictions)[-3:][::-1]

    for i in top_3:
        st.write(f"{class_names[i]}: {predictions[i]*100:.2f}%")

    if confidence < 0.5:
        st.warning("⚠️ Low confidence prediction. Try a clearer image.")
