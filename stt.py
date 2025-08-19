import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load Whisper model + processor from Hugging Face
model_name = "openai/whisper-large-v3"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

st.title("🎙️ Whisper STT (Transformers) Demo")

duration = st.slider("Recording duration (sec)", 3, 10, 5)
if st.button("Start Recording"):
    st.info("Recording...")
    fs = 16000  # Whisper expects 16kHz
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    st.success("Recording complete!")

    st.info("Transcribing...")
    input_features = processor(audio[:, 0], sampling_rate=16000,padding = 'longest',truncation = False, return_tensors="pt", return_attention_mask = True).input_features
    predicted_ids = model.generate(input_features, language = 'en', task = 'translate', attention_mask = input_features.attention_mask)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    st.success("Transcription complete!")
    st.write("**Transcribed Text:**")
    st.write(transcription)