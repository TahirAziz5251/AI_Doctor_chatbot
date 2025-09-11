import os
import gradio as gr

from main import encode_image, analyze_image_with_query
from patient_voice import transcribe_with_groq, record_audio
from doctor_voice import play_audio, text_to_speech_with_gtts, text_to_speech_with_elevenlabs

system_prompt = """ You have to act a professional doctor. You will help patients by analyzing their medical images and answering their questions based on the images and their voice input. Provide clear, concise, and empathetic responses.
If you make a differential, suggest some remedies for them. Don't add any number and special characters in your response. Don't say
'In the image I see ' but say "with what i see , I think you have ....'
Don't response as an AI model in markdown, your answer should be mimic that of an actual doctor not an AI bot, keep your 
answer concise (max 2 sentence). No preamle, start your answer right way please."""

def process_input(audio_file_path, image_path):
    # Speech to Text
    if audio_file_path:
        audio_file_path = "patient_voice.mp3"
        record_audio(file_path=audio_file_path, timeout=15, phrase_time_limit=None)

        speech_to_text_output = transcribe_with_groq(
            key=os.environ.get("GROQ_API_KEY"),
            audio_file_path=audio_file_path,
            stt_model="whisper-large-v3"
        )

        if isinstance(speech_to_text_output, tuple):
            speech_to_text_output = speech_to_text_output[0]
    else:
        speech_to_text_output = "No audio provided."


    # --- Image Analysis with Query ---
    if image_path:
        doctor_response = analyze_image_with_query(
            query=system_prompt+ speech_to_text_output,
            encoded_image=encode_image(image_path),
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
    else:
        doctor_response = "No image provided for analysis."

    # --- Text to Speech ---
    output_audio_file = "final.mp3"
    text_to_speech_with_elevenlabs(
        input_text = doctor_response,
        output_file=output_audio_file
    )
    play_audio("final.mp3")
    try: 
        play_audio(output_audio_file)
    except Exception as e:
        print(f"Could not play audio: {e}")

    return speech_to_text_output, doctor_response, output_audio_file


# --- Gradio Interface ---
iface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Audio(sources="microphone", type="filepath", label="Patient Voice Input"),
        gr.Image(type="filepath", label="Upload Medical Image")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(label="Doctor's Voice", type="filepath")
    ],
    title="AI Doctor Chatbot with Vision and Voice",
)

if __name__ == "__main__":
    iface.launch(debug=True)