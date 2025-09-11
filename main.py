import os
from dotenv import load_dotenv
from PIL import Image
import io, base64
from groq import Groq

load_dotenv()
key = os.getenv("GROQ_API_KEY")



#  Compress + Encode Image 
# image_path = "applying-moisturizer-skin-with-psoriasis.jpg"

def encode_image(image_path):
    
    img = Image.open(image_path)

    # resize to max 512x512 to reduce size
    img.thumbnail((512, 512))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=70)  # lower quality to reduce size
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

query = "Is there something wrong with this skin?"
model = "meta-llama/llama-4-scout-17b-16e-instruct"

def analyze_image_with_query(query, model, encoded_image):
    client = Groq(api_key=key)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text",
                    "text": query},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }
            ],
        }
    ]

    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=512,   # reduce tokens for safety
        temperature=0.7,
    )

    return chat_completion.choices[0].message.content

    
# Groq Client 

