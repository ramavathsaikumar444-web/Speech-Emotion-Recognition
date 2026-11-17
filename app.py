
import streamlit as st
import librosa
import librosa.display
import numpy as np
import pandas as pd
import joblib
import tempfile
import os
import matplotlib.pyplot as plt


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Speech Emotion AI",
    page_icon="🎙️",
    layout="wide"
)


# LOAD MODEL

@st.cache_resource
def load_model():
    data = joblib.load("emotion_recognition_model.pkl")
    return (
        data["model"],
        data["feature_columns"],
        data["emotion_names"]
    )


model, feature_columns, emotion_names = load_model()


# EMOTION NAME

def get_emotion_name(prediction):
    if isinstance(emotion_names, dict):
        return emotion_names.get(prediction, str(prediction))

    if isinstance(emotion_names, list):
        try:
            return emotion_names[int(prediction)]
        except (ValueError, IndexError, TypeError):
            return str(prediction)

    return str(prediction)


# FEATURE EXTRACTION

def extract_features(file_path):

    audio, sr = librosa.load(file_path, sr=None)

    audio, _ = librosa.effects.trim(audio, top_db=25)

    if len(audio) == 0:
        raise ValueError("No usable audio detected.")

    audio = librosa.util.normalize(audio)

    features = []

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512
    )

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    pitch, _, _ = librosa.pyin(
        audio,
        fmin=65,
        fmax=400,
        sr=sr,
        frame_length=2048,
        hop_length=512
    )

    pitch = pitch[~np.isnan(pitch)]

    if len(pitch) > 0:
        features.extend([np.mean(pitch), np.std(pitch)])
    else:
        features.extend([0, 0])

    rms = librosa.feature.rms(y=audio)[0]
    features.extend([np.mean(rms), np.std(rms)])

    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features.extend([np.mean(zcr), np.std(zcr)])

    return np.array(features)


# DESIGN

st.markdown("""
<style>
.stApp {
    background: #0b1220;
    color: #e8edf5;
}

.block-container {
    max-width: 1050px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #f1f5f9 !important;
}

.subtitle {
    color: #94a3b8;
    font-size: 16px;
    margin-top: -12px;
    margin-bottom: 30px;
}

.result-box {
    background: #14243a;
    border: 1px solid #294563;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
    margin-top: 20px;
}

.result-label {
    color: #94a3b8;
    font-size: 15px;
}

.result-value {
    color: #60a5fa;
    font-size: 36px;
    font-weight: 700;
    margin-top: 8px;
}

div.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 8px;
    height: 48px;
    font-size: 16px;
    font-weight: 600;
}

div.stButton > button:hover {
    background: #1d4ed8;
    color: white;
    border: none;
}

[data-testid="stFileUploader"] {
    background: #14243a;
    border: 1px solid #294563;
    border-radius: 10px;
    padding: 15px;
}

hr {
    border-color: #25364d;
}
</style>
""", unsafe_allow_html=True)


# HEADER

st.title("Speech Emotion AI")

st.markdown(
    '<div class="subtitle">'
    'Speech emotion recognition using machine learning'
    '</div>',
    unsafe_allow_html=True
)


# AUDIO UPLOAD

st.subheader("Upload Audio")

uploaded_file = st.file_uploader(
    "Select an audio recording",
    type=["wav", "mp3", "ogg"]
)


if uploaded_file is not None:

    st.audio(uploaded_file)

    temp_path = None

    try:
        suffix = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        # LOAD AUDIO FOR DISPLAY

        display_audio, sample_rate = librosa.load(
            temp_path,
            sr=None,
            mono=False
        )

        if display_audio.ndim == 2:
            mono_audio = np.mean(display_audio, axis=0)
            channels = display_audio.shape[0]
        else:
            mono_audio = display_audio
            channels = 1

        duration = len(mono_audio) / sample_rate

        # AUDIO INFORMATION

        st.caption(
            f"Duration: {duration:.2f} seconds  |  "
            f"Sample rate: {sample_rate:,} Hz  |  "
            f"Channels: {channels}"
        )

        # EMOTION PREDICTION

        st.divider()
        st.subheader("Emotion Detection")

        if st.button("Analyze Emotion", use_container_width=True):

            with st.spinner("Analyzing audio..."):

                extracted_features = extract_features(temp_path)

                if len(extracted_features) != len(feature_columns):
                    st.error("Feature count does not match the trained model.")

                else:
                    input_data = pd.DataFrame(
                        [extracted_features],
                        columns=feature_columns
                    )

                    prediction = model.predict(input_data)[0]
                    emotion = get_emotion_name(prediction)

                    st.markdown(
                        f"""
                        <div class="result-box">
                            <div class="result-label">Detected Emotion</div>
                            <div class="result-value">{emotion}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # PROBABILITY

                    if hasattr(model, "predict_proba"):

                        probabilities = model.predict_proba(input_data)[0]
                        classes = model.classes_

                        probability_df = pd.DataFrame({
                            "Emotion": [
                                get_emotion_name(label)
                                for label in classes
                            ],
                            "Probability": probabilities
                        })

                        probability_df = probability_df.sort_values(
                            "Probability",
                            ascending=False
                        )

                        st.subheader("Emotion Distribution")

                        st.bar_chart(
                            probability_df.set_index("Emotion")
                        )

                        st.caption(
                            "Estimated model probabilities; "
                            "these may not be calibrated confidence scores."
                        )

                    else:
                        st.info(
                            "Emotion probability chart requires a model "
                            "trained with SVM probability=True."
                        )

        # OPTIONAL AUDIO ANALYSIS

        st.divider()

        with st.expander("View Audio Analysis"):

            st.subheader("Waveform")

            fig, ax = plt.subplots(figsize=(11, 2.5))

            time_axis = np.arange(len(mono_audio)) / sample_rate

            ax.plot(time_axis, mono_audio, linewidth=0.6)
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude")
            ax.grid(alpha=0.2)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("MFCC")

            mfcc = librosa.feature.mfcc(
                y=mono_audio,
                sr=sample_rate,
                n_mfcc=13,
                n_fft=2048,
                hop_length=512
            )

            fig, ax = plt.subplots(figsize=(11, 3))

            img = librosa.display.specshow(
                mfcc,
                sr=sample_rate,
                hop_length=512,
                x_axis="time",
                ax=ax
            )

            ax.set_ylabel("MFCC")
            fig.colorbar(img, ax=ax)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Pitch")

            pitch, _, _ = librosa.pyin(
                mono_audio,
                fmin=65,
                fmax=400,
                sr=sample_rate,
                frame_length=2048,
                hop_length=512
            )

            pitch_time = librosa.times_like(
                pitch,
                sr=sample_rate,
                hop_length=512
            )

            fig, ax = plt.subplots(figsize=(11, 2.5))

            ax.plot(pitch_time, pitch, linewidth=0.8)
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Frequency (Hz)")
            ax.grid(alpha=0.2)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Energy")

            rms = librosa.feature.rms(y=mono_audio)[0]

            rms_time = librosa.frames_to_time(
                np.arange(len(rms)),
                sr=sample_rate
            )

            fig, ax = plt.subplots(figsize=(11, 2.5))

            ax.plot(rms_time, rms, linewidth=0.8)
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Energy")
            ax.grid(alpha=0.2)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    except Exception as e:
        st.error(f"Unable to process audio: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

else:
    st.info("Upload an audio recording to begin.")

# FOOTER

st.divider()

st.caption("Speech Emotion AI | Machine Learning Project")
