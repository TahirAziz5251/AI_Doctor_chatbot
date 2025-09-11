import os
from dotenv import load_dotenv
from pydub import AudioSegment
from gtts import gTTS
import platform
import subprocess
import elevenlabs
from elevenlabs import ElevenLabs

load_dotenv()
key = os.getenv("ELEVENLABS_API_KEY")

#  Playback helper 
def play_audio(file_path):
    os_name = platform.system()
    try:
        if os_name == "Darwin":  
            subprocess.run(["afplay", file_path])
        elif os_name == "Windows":  # Windows (convert to WAV if mp3)
            if file_path.endswith(".mp3"):
                wav_file = file_path.replace(".mp3", ".wav")
                sound = AudioSegment.from_mp3(file_path)
                sound.export(wav_file, format="wav")
                file_path = wav_file
            subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{file_path}').PlaySync();"])
        elif os_name == "Linux":  # Linux
            subprocess.run(["aplay", file_path])
        else:
            raise Exception("Unsupported OS")
    except Exception as e:
        print(f"Could not play audio: {e}")


#  gTTS 
def text_to_speech_with_gtts(input_text, output_file):
    language = 'en'
    audioobj = gTTS(text=input_text, lang=language, slow=False)
    audioobj.save(output_file)
    play_audio(output_file)


#  ElevenLabs 
def text_to_speech_with_elevenlabs(input_text, output_file):
    client = ElevenLabs(api_key=key)
    audio = client.text_to_speech.convert(
        voice_id="gs0tAILXbY5DNrJrsM6F",  # Example voice ID
        model_id="eleven_turbo_v2",
        output_format="mp3_22050_32",
        text=input_text
    )
    elevenlabs.save(audio, output_file)
    play_audio(output_file)

if __name__ == "__main__":
    input_text = "Hello, I am your AI Doctor Assistant. How can I help you today?"

    # gTTS (autoplay)
    text_to_speech_with_gtts(input_text, output_file="doctor_voice_gtts.mp3")

    # ElevenLabs (autoplay)
    text_to_speech_with_elevenlabs(input_text, output_file="doctor_voice_elevenlabs.mp3")
