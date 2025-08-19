import os
from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyA_lZ78Rf_J9lCBqpu4hFaHSzYopB4CY0Y")


# Read the audio file as bytes
with open('sessions/session_20250806_193505.wav', 'rb') as f:
    audio_bytes = f.read()

# Send the request to the model
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        'Transcribe this audio clip',
        types.Part.from_bytes(
            data=audio_bytes,
            mime_type='audio/wav',
        )
    ]
)

# Print the transcribed text
print(response.text)