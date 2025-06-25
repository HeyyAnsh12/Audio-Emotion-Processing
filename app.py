import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os
from keras.models import load_model

# Load pre-trained model from local directory
emotion_model = load_model("emotion_model.h5") 

# Categories of emotions (edit to match model training)
EMOTION_CLASSES = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def compute_mfcc(audio_path):
    """
    Extracts MFCC features from an audio file.
    Uses 40 coefficients averaged over time.
    """
    signal, rate = librosa.load(audio_path, duration=3, offset=0.5)
    mfcc_data = librosa.feature.mfcc(y=signal, sr=rate, n_mfcc=40)
    averaged_mfcc = np.mean(mfcc_data.T, axis=0)
    return averaged_mfcc

def main():
    st.header("Audio-Based Emotion Identifier")
    st.write("Submit a `.wav` file and receive an emotion classification.")

    input_audio = st.file_uploader("Upload a WAV audio file", type=["wav"])

    if input_audio:
        st.audio(input_audio, format="audio/wav")

        # Temporary storage of uploaded audio
        with open("buffered.wav", "wb") as temp_audio:
            temp_audio.write(input_audio.read())

        # Audio -> Features -> Prediction
        extracted = compute_mfcc("buffered.wav")
        input_features = np.expand_dims(extracted, axis=0)

        probabilities = emotion_model.predict(input_features)
        emotion = EMOTION_CLASSES[np.argmax(probabilities)]

        st.subheader("Classification Output")
        st.success(f"Detected Emotion: **{emotion.upper()}**")

        # Cleanup temporary audio
        os.remove("buffered.wav")

if __name__ == "__main__":
    main()
