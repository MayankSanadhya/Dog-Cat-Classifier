import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def import_model(x):
    classifier = tf.keras.models.load_model(x)
    return classifier
model = import_model('facefeatures_new_model.h5')
st.write("""
          # Dog Cat Prediction
         """
        )
file = st.file_uploader("Please upload an Dog or Cat image",type=['jpg'])
import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    
    size = (224,224)
    image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    
    return prediction
if file is None:
    st.text("Please Upload an image file")
else:
    image = Image.open(file)
    st.image(image,use_column_width=True)
    predictions = import_and_predict(image,model)
    if result[0][0] == 1:     # [0][0]  first zero is for batch and second zero is for singleimage.
        prediction = 'dog'
    else:
        prediction = 'cat'
    string = "This image most likely is : "+ prediction
    st.success(string)
