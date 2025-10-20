from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from twilio.twiml.voice_response import VoiceResponse, Gather
from dotenv import load_dotenv
from openai import OpenAI
import requests, os, uuid, traceback
from session_store import get_history, set_history, clear_history
import json

CHANGE_KEYWORDS = {"change", "modify", "edit", "add", "remove", "cancel"}

def wants_change(speech: str) -> bool:
    if not speech:
        return False
    s = speech.lower()
    return any(k in s for k in CHANGE_KEYWORDS)




def parse_agent_json(text: str):
    """
    Try to parse the model output as JSON. Expected shape:
    {
      "cart": [{"item": "...", "qty": 1, "size": "...", "sides": [], "sauces": []}],
      "customer_name": "optional",
      "action": "continue|confirm|finalize",
      "say_text": "short sentence to speak next"
    }
    """
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return None
        # Minimal validation
        if "say_text" not in data or "action" not in data:
            return None
        return data
    except Exception:
        return None

# Load from file named 'env'
load_dotenv("env", override=True)

OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY   = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID  = os.getenv("ELEVENLABS_VOICE_ID")
PUBLIC_BASE_URL      = os.getenv("PUBLIC_BASE_URL")   # optional full URL, e.g. https://yourhost.ngrok-free.app
NGROK_DOMAIN         = os.getenv("NGROK_DOMAIN")      # optional host only, e.g. yourhost.ngrok-free.app
ENV                  = os.getenv("ENV", "dev")        # dev|prod

# Fail fast on critical envs
missing = [k for k, v in {
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "ELEVENLABS_API_KEY": ELEVENLABS_API_KEY,
    "ELEVENLABS_VOICE_ID": ELEVENLABS_VOICE_ID,
}.items() if not v]
if missing:
    raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

# Ensure audio dir exists BEFORE mounting StaticFiles
os.makedirs("audio", exist_ok=True)

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

# Serve /audio/*
app.mount("/audio", StaticFiles(directory="audio"), name="audio")

def get_base_url(request: Request) -> str:
    """
    Resolve the public base URL for Twilio to call back / fetch audio.
    Priority:
      1) PUBLIC_BASE_URL (full https URL)
      2) NGROK_DOMAIN (host only, assume https)
      3) X-Forwarded headers or Host header
    """
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL.rstrip("/")
    if NGROK_DOMAIN:
        return f"https://{NGROK_DOMAIN}".rstrip("/")
    proto = request.headers.get("x-forwarded-proto", "https")
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    return f"{proto}://{host}".rstrip("/")

def text_to_speech_elevenlabs(text: str) -> str:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    body = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "optimize_streaming_latency": 1
    }

    r = requests.post(url, json=body, headers=headers, timeout=30)
    if r.status_code != 200 or not r.content:
        raise RuntimeError(f"ElevenLabs TTS failed: {r.status_code} {r.text[:200]}")
    file_id = str(uuid.uuid4())
    mp3_path = f"audio/{file_id}.mp3"
    with open(mp3_path, "wb") as f:
        f.write(r.content)
    return mp3_path

def chatgpt_reply(prompt, history):
    system_prompt = {
        "role": "system",
        "content": (
            "You are OrderPilot, a professional phone agent for Captain Sam’s Fish & Chicken.\n"
            "Only take food pickup orders. Keep answers short and precise. No greetings.\n"
            "Output must be STRICT JSON with keys: cart, customer_name, action, say_text.\n"
            "Schema:\n"
            "{\n"
            '  "cart": [\n'
            '    {"item": "3pc Fish Dinner", "qty": 1, "size": "large", "sides": ["fries"], "sauces": ["tartar"]}\n'
            "  ],\n"
            '  "customer_name": "optional string",\n'
            '  "action": "continue|confirm|finalize",\n'
            '  "say_text": "Short sentence to speak next (<= 200 chars)."\n'
            "}\n"
            "Rules:\n"
            "- Never suggest items unless the customer asks.\n"
            "- If customer mentions an item, confirm size/sauce/side briefly.\n"
            "- For confirm step: repeat order once, ask once if anything else.\n"
            "- For finalize: return action=\"finalize\" and include a brief say_text like "
            '"Your order is confirmed. Please pick up in 15–20 minutes."\n'
            "Respond with JSON only. No extra text."
        )
    }
    messages = [system_prompt] + history + [{"role": "user", "content": prompt}]
    # Model can be updated if you prefer a different one
    resp = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    response_format={"type": "json_object"}  # <- add this line
)
    reply = resp.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return reply, messages

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.api_route("/voice", methods=["GET", "POST"])
async def voice(request: Request):
    base = get_base_url(request)
    vr = VoiceResponse()

    # Play pre-generated welcome if present; otherwise safe fallback
    welcome_path = "audio/welcome.mp3"
    if os.path.exists(welcome_path) and os.path.getsize(welcome_path) > 0:
        vr.play(f"{base}/audio/welcome.mp3")
    else:
        vr.say("Hey there! Thanks for calling Captain Sam’s Fish and Chicken. "
               "Want to place an order for pickup?")

    gather = Gather(
        input="speech",
        action=f"{base}/gather",
        method="POST",
        speech_timeout="3",
        timeout=10
    )
    vr.append(gather)

    # If no speech captured, hit /gather again via POST
    vr.redirect(f"{base}/gather", method="POST")
    return Response(content=str(vr), media_type="application/xml")

@app.api_route("/gather", methods=["GET", "POST"])
async def gather(request: Request):
    try:
        base = get_base_url(request)
        form = await request.form()
        call_sid = form.get("CallSid", "unknown")
        user_input = (form.get("SpeechResult") or "").strip()

        vr = VoiceResponse()

        if not user_input:
            # Silence / no STT result → re-gather
            gather = Gather(
                input="speech",
                action=f"{base}/gather",
                method="POST",
                speech_timeout="3",
                timeout=10
            )
            vr.append(gather)
            return Response(content=str(vr), media_type="application/xml")

        # Build conversation using Redis-backed history
        history = get_history(call_sid)
        try:
            chat_response, updated_history = chatgpt_reply(user_input, history)
        except Exception as ai_err:
            if ENV != "prod":
                print("OpenAI error:", ai_err)
                traceback.print_exc()
            chat_response = "Sorry, I had trouble. Please say your order again with the item and size."
            updated_history = history  # keep prior history unchanged

        # Parse model output -> say_text and action
        agent = parse_agent_json(chat_response)
        if agent and isinstance(agent.get("say_text"), str):
            speak_text = agent["say_text"].strip()
        else:
            speak_text = chat_response.strip()

        # Enforce length cap (still good practice)
        MAX_TTS = 300
        if len(speak_text) > MAX_TTS:
            trimmed = speak_text[:MAX_TTS].rsplit(".", 1)[0].strip()
            speak_text = (trimmed + ".") if trimmed else speak_text[:MAX_TTS]

        # Persist updated history once
        set_history(call_sid, updated_history)

        # Speak the response (TTS preferred, fallback to <Say>)
        try:
            mp3_path = text_to_speech_elevenlabs(speak_text)
            audio_url = f"{base}/audio/{os.path.basename(mp3_path)}"
            vr.play(audio_url)
        except Exception as tts_err:
            if ENV != "prod":
                print("ElevenLabs error:", tts_err)
                traceback.print_exc()
            vr.say("Sorry, I had trouble generating audio. Here is the response.")
            vr.say(speak_text)

        # If the agent finalized, give a short change-your-mind window
        if agent and agent.get("action") == "finalize":
            gather = Gather(
                input="speech",
                action=f"{base}/finalize_check",
                method="POST",
                speech_timeout="2",  # end after ~2s of silence
                timeout=3            # hard cap
            )
            vr.append(gather)
            # No redirect here; /finalize_check will decide to hang up or continue
            return Response(content=str(vr), media_type="application/xml")

        # Otherwise, continue the normal loop
        vr.redirect(f"{base}/gather", method="POST")
        return Response(content=str(vr), media_type="application/xml")

    except Exception as e:
        # Never return a 500 to Twilio; send valid TwiML instead
        if ENV != "prod":
            print("Fatal /gather error:", e)
            traceback.print_exc()
        vr = VoiceResponse()
        vr.say("Sorry, there was an error. Please say your order again.")
        vr.redirect("/voice", method="POST")
        return Response(content=str(vr), media_type="application/xml", status_code=200)

@app.api_route("/finalize_check", methods=["POST"])
async def finalize_check(request: Request):
    base = get_base_url(request)
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    said = (form.get("SpeechResult") or "").strip()

    vr = VoiceResponse()

    if wants_change(said):
        vr.say("No problem — what would you like to change?")
        vr.redirect(f"{base}/gather", method="POST")
        return Response(content=str(vr), media_type="application/xml")

    # Otherwise pause briefly, say goodbye, clear session, and hang up
    vr.pause(length=1)
    vr.say("Goodbye.")
    try:
        clear_history(call_sid)
    except Exception:
        pass
    vr.hangup()
    return Response(content=str(vr), media_type="application/xml")
