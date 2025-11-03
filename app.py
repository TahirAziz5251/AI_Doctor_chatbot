import os
import gradio as gr

from main import encode_image, analyze_image_with_query
from patient_voice import transcribe_with_groq, record_audio
from doctor_voice import text_to_speech_with_elevenlabs


system_prompt = """ You have to act a professional doctor. You will help patients by analyzing their medical images and answering their questions based on the images and their voice input. Provide clear, concise, and empathetic responses.
If you make a differential, suggest some remedies for them. Don't add any number and special characters in your response. Don't say
'In the image I see ' but say "with what i see , I think you have ....'
Don't response as an AI model in markdown, your answer should be mimic that of an actual doctor not an AI bot, keep your 
answer concise (max 2 sentence). No preamle, start your answer right way please."""

def process_input(audio_file_path, image_path):
    # --- Patient Voice to Text ---
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

    # --- Image + Query Analysis ---
    if image_path:
        doctor_response = analyze_image_with_query(
            query=system_prompt + speech_to_text_output,
            encoded_image=encode_image(image_path),
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
    else:
        doctor_response = "No image provided for analysis."

    # --- Doctor Voice (TTS) ---
    output_audio_file = "final.mp3"
    text_to_speech_with_elevenlabs(
        input_text=doctor_response,
        output_file=output_audio_file
    )

    return speech_to_text_output, doctor_response, output_audio_file


def toggle_audio(is_playing, audio_path):
    """Custom Play/Stop toggle button"""
    if is_playing:
        # Stop audio
        return None, False, "▶️ Play Doctor Response"
    else:
        # Start audio
        return audio_path, True, "⏹️ Stop Doctor Response"


# --- Gradio Interface ---
with gr.Blocks(title="AI Doctor") as iface:
    with gr.Row():
        patient_audio = gr.Audio(sources="microphone", type="filepath", label="Patient Voice Input")
        medical_image = gr.Image(type="filepath", label="Upload Medical Image")

    stt_text = gr.Textbox(label="Speech to Text", lines=3, interactive=False)
    doctor_text = gr.Textbox(label="Doctor's Response", lines=8, interactive=False)

    # Hidden Audio player (not autoplay)
    doctor_audio = gr.Audio(label="Doctor's Voice", type="filepath", interactive=False)

    # State to track play/stop
    is_playing = gr.State(False)

    # Custom Play/Stop Button
    play_button = gr.Button("▶️ Play Doctor Response")

    # Run analysis
    run_btn = gr.Button("Analyze & Generate Response")

    # Outputs from main function
    run_btn.click(
        fn=process_input,
        inputs=[patient_audio, medical_image],
        outputs=[stt_text, doctor_text, doctor_audio]
    )

    # Toggle button
    play_button.click(
        fn=toggle_audio,
        inputs=[is_playing, doctor_audio],
        outputs=[doctor_audio, is_playing, play_button]
    )

if __name__ == "__main__":
    iface.launch(show_api=False, share=True, debug=True)
