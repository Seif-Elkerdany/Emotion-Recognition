# Import used modules
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Load the model and make dictionary with the prediction labels
# model = tf.keras.models.load_model(r"CNN_FER13_Model_zip.zip")
model = tf.keras.layers.TFSMLayer("CNN_FER13_Model.keras")
emotion_dic= {0: 'Angry', 1: 'Fearful', 2: 'Happy', 3: 'Surprised', 4: 'Neutral', 5: 'Sad'}

@st.cache_data
def load_image(image_file):
    # Reading the image using PIL
    img = Image.open(image_file).convert('RGB')
    return img

def main():
    st.title("Emotion Recoginition")

    menu = ["Webcam", "Upload an Image"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Webcam":
        st.subheader("Classifing by webcam video")
        
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)

        while choice == "Webcam":
            # Capture frame-by-frame
            ret, frame = camera.read()
            frame = cv2.resize(frame, (1280, 720))
            # Haarcascades cv2 classifier that helps to detect a face
            face_detector = cv2.CascadeClassifier(r"haarcascades\haarcascade_frontalface_default.xml")
            # Make frames grey before entering the model
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Number of faces can Haarcascades detect at a time
            num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor= 1.3, minNeighbors=2)
            
            # Drawing a box around the 
            for(x, y, w, h) in num_faces:
                # Crop (48, 48) image from the fram to predict its label
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0,255,0), 4)
                roi_gray_frame = gray_frame[y:y+h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1 ), 0)
                # Remove any possible noise and raise the brightness up 
                blur = cv2.GaussianBlur(cropped_img, (1, 1), 0)
                white = cv2.convertScaleAbs(blur, beta = 120)
                # Predection stage
                emotion_pred = model.predict(white)
                maxindex = int(np.argmax(emotion_pred))
                # Drawing a box around the face and write the prediction on top of the rectangle
                cv2.putText(frame, emotion_dic[maxindex], (x + 5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4 )
            # Return colored frame to display it
            final_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Update the image in Streamlit
            FRAME_WINDOW.image(final_frame) 
            
    elif choice == "Upload an Image":
        st.subheader("Classifing by Image")
        # Upload the image to streamlit 
        img_file = st.file_uploader('Upload the image here', type=['png', 'PNG', 'JPG', 'jpg', 'jpeg', 'JPEG'])
        
        if img_file is not None:
            # Display the image 
            st.image(load_image(img_file))
            # Convert the image into opencv format 
            img = cv2.cvtColor(np.array(load_image(img_file)), cv2.COLOR_RGB2BGR)
            
            face_detector = cv2.CascadeClassifier(r"haarcascades\haarcascade_frontalface_default.xml")
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            num_faces = face_detector.detectMultiScale(gray_img, scaleFactor= 1.3, minNeighbors=2)
            
            for(x, y, w, h) in num_faces:
                cv2.rectangle(img, (x, y - 50), (x + w, y + h + 10), (0,255,0), 4)
                roi_gray_img = gray_img[y:y+h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_img, (48, 48)), -1 ), 0)
                
                blur = cv2.GaussianBlur(cropped_img, (1, 1), 0)
                white = cv2.convertScaleAbs(blur, beta = 120)
                
                emotion_pred = model.predict(white)
                maxindex = int(np.argmax(emotion_pred))
                
                img = cv2.putText(img, emotion_dic[maxindex], (x + 5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4 )
                
            final_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            st.write("---")
            st.title("Prediction")
            st.image(final_img, use_column_width=True)
                
    st.markdown("Developed by: Seif_Elkerdany© & Mohamed_Walied© 2024", unsafe_allow_html=True)
    
if __name__ == '__main__':
    main()
