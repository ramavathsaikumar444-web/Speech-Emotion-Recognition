
import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
import tempfile
import os

# --------------------------------
# 1. Load trained model
# --------------------------------

@st.cache_resource
def load_model():
    data = joblib.load("emotion_recognition_model.pkl")

    model = data["model"]
    feature_columns = data["feature_columns"]
    emotion_names = data["emotion_names"]

    return model, feature_columns, emotion_names


model, feature_columns, emotion_names = load_model()


# --------------------------------
# 2. Feature Extraction
# --------------------------------

def extract_features(file_path):

    # Load audio
    y, sr = librosa.load(file_path, sr=None)

    # Preprocessing
    y, _ = librosa.effects.trim(y, top_db=25)
    y = librosa.util.normalize(y)

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Pitch
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=65,
        fmax=400,
        sr=sr,
        frame_length=2048,
        hop_length=512
    )

    voiced_pitch = f0[voiced_flag & np.isfinite(f0)]

    if len(voiced_pitch) > 0:
        pitch_mean = np.mean(voiced_pitch)
        pitch_std = np.std(voiced_pitch)
    else:
        pitch_mean = 0.0
        pitch_std = 0.0

    # Energy
    energy = librosa.feature.rms(
        y=y,
        frame_length=2048,
        hop_length=512
    )

    energy_mean = np.mean(energy)
    energy_std = np.std(energy)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=2048,
        hop_length=512
    )

    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # Combine features
    return np.concatenate([
        mfcc_mean,
        mfcc_std,
        [pitch_mean, pitch_std],
        [energy_mean, energy_std],
        [zcr_mean, zcr_std]
    ])


# --------------------------------
# 3. Emotion Name Mapping
# --------------------------------

def get_emotion_name(prediction):

    if isinstance(emotion_names, dict):
        return emotion_names.get(
            prediction,
            emotion_names.get(str(prediction), str(prediction))
        )

    return emotion_names[int(prediction) - 1]


# --------------------------------
# 4. Streamlit Interface
# --------------------------------

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Speech Emotion Recognition")

st.write(
    "Upload a speech audio file and our trained "
    "SVM model will predict the emotion."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload your audio file",
    type=["wav", "mp3", "ogg"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    if st.button("Predict Emotion", type="primary"):

        temp_path = None

        try:
            # Save uploaded audio temporarily
            suffix = os.path.splitext(uploaded_file.name)[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            # Extract features
            features = extract_features(temp_path)

            # Prepare input DataFrame
            if len(features) != len(feature_columns):
                raise ValueError(
                    f"Expected {len(feature_columns)} features, "
                    f"but extracted {len(features)}."
                )

            input_data = pd.DataFrame(
                [features],
                columns=feature_columns
            )

            # Predict emotion
            prediction = model.predict(input_data)[0]

            emotion = get_emotion_name(prediction)

            st.success("Emotion Prediction Completed!")

            st.subheader("Predicted Emotion")

            st.markdown(
                f"## 🎯 {emotion}"
            )

        except Exception as e:
            st.error(f"Error during prediction: {e}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

st.divider()

st.caption(
    "Speech Emotion Recognition | "
    "MFCC + Pitch + Energy + ZCR + SVM"
)
