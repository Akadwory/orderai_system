import os
import requests
from dotenv import load_dotenv

# Load from file named 'env'
load_dotenv("env")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")


def generate_intro_audio():
    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        raise RuntimeError("Missing ELEVENLABS_API_KEY or ELEVENLABS_VOICE_ID in env")

    text = (
        "Hey there! Thanks for calling Captain Sam’s Fish & Chicken. "
        "Want to place an order for pickup?"
    )
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    body = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }

    r = requests.post(url, json=body, headers=headers, timeout=30)
    if r.status_code != 200 or not r.content:
        try:
            err = r.json()
        except Exception:
            err = {"raw": r.text}
        raise RuntimeError(f"ElevenLabs welcome TTS failed: {r.status_code} {err}")

    os.makedirs("audio", exist_ok=True)
    out = "audio/welcome.mp3"
    with open(out, "wb") as f:
        f.write(r.content)

    print("Welcome message saved to", out)


if __name__ == "__main__":
    generate_intro_audio()
