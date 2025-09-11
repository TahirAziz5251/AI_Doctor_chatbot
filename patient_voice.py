import os
import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio(file_path, timeout=15, phrase_time_limit=None):
    """
    Records audio from the microphone and saves it as an MP3 file.
    
    Args:
        file_path (str): The path where the  recorded audio file will be saved.
        timeout (int): Maximum seconds to wait for a phrase to start.
        phrase_time_limit (int): Maximum seconds for a phrase to be recorded (in seconds)."""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Listening for audio input...")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete. Saving audio...")

            wave_data = audio.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wave_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")

            logging.info(f"Audio saved to {file_path}")
    except Exception as e:
        logging.error(f"An error occurred while recording audio: {e}")
audio_file_path = "patient_voice.mp3"
record_audio(file_path=audio_file_path)


from dotenv import load_dotenv
from groq import Groq
load_dotenv()

key = os.getenv("GROQ_API_KEY")
stt_model = "whisper-large-v3"

def transcribe_with_groq(audio_file_path, stt_model, key):
    client = Groq(api_key=key)
    
    audio_file = open(audio_file_path, "rb")
    transcription = client.audio.transcriptions.create(
        model=stt_model,
        file=audio_file,
        language= "en"
    )
    return "Transcription:", transcription.text