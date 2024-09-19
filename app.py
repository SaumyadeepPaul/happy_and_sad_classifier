import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your model
model = tf.keras.models.load_model('happy_sad_cnn_model.h5')

# Streamlit UI
st.title("Happy or Sad Classifier")
uploaded_file = st.file_uploader("Choose an image...")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    img = np.array(image.resize((64, 64))) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    result = "Happy" if prediction[0][0] < 0.5 else "Sad"
    st.write(result)
