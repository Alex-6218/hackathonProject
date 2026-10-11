import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from elevenlabs import (
    AudioFormat,
    CommitStrategy,
    ElevenLabs,
    RealtimeAudioOptions,
    RealtimeEvents,
)
import sounddevice as sd
from scipy.io import wavfile

from dotenv import load_dotenv

import os, base64, time, asyncio, threading, cv2, queue, re

load_dotenv() 

#TRANSCRIPTION STUFF
last_word = ""
last_word_t = 0.0

SAMPLE_RATE = 48000
BLOCKSIZE = 4800  # 100 ms per block

def clean(text):
    words = text.strip().split()
    return words[-1].strip(".,!?;:").lower() if words else ""


async def transcribe_forever():
    global last_word
    loop = asyncio.get_running_loop()
    audio_q = asyncio.Queue()          # mic thread -> this loop
    session_ready = asyncio.Event()

    

    #create client & connection
    client = ElevenLabs(api_key=os.getenv("API_KEY"))
    connection = await client.speech_to_text.realtime.connect(
        RealtimeAudioOptions(
            model_id="scribe_v2_realtime",
            audio_format=AudioFormat.PCM_48000,
            sample_rate=48000,
            commit_strategy=CommitStrategy.VAD,
        )
    )

    #define event handlers
    def on_session_started(data):
        print("[stt] session started")
        session_ready.set()

    def on_transcript(data):
        global last_word, last_word_t
        word = clean(data.get("text", ""))
        if word:
            last_word = word
            last_word_t = time.monotonic()
    
    def on_error(error):
        print(f"[stt error] {error}")
 
    def on_close():
        print("[stt] connection closed")

    #activate event handlers
    connection.on(RealtimeEvents.SESSION_STARTED, on_session_started)
    connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, on_transcript)
    connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, on_transcript)
    connection.on(RealtimeEvents.ERROR, on_error)
    connection.on(RealtimeEvents.CLOSE, on_close)

    def audio_callback(indata, frames, time_info, status):
        loop.call_soon_threadsafe(audio_q.put_nowait, bytes(indata))
 
    # grab mic input and stream it to the server
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16",
        blocksize=BLOCKSIZE,
        device=4,                 # index or name substring
        callback=audio_callback,
    ):
        await session_ready.wait()
        while True:
            chunk = await audio_q.get()
            await connection.send({
                "audio_base_64": base64.b64encode(chunk).decode("ascii"),
                "sample_rate": SAMPLE_RATE,
            })
 
 
def start_transcriber():
    def runner():
        while True:
            try:
                asyncio.run(transcribe_forever())
            except Exception as e:
                print(f"[stt] crashed: {e}; retrying in 2s")
                time.sleep(2)
    threading.Thread(target=runner, daemon=True).start()
 
start_transcriber()
 



#GESTURE STUFF
#define camera object and set window size for display
cam = cv2.VideoCapture(2)
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))	

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

#ai hand stuff
#create gesture recognizer object
options = vision.GestureRecognizerOptions(
    base_options=python.BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
recognizer = vision.GestureRecognizer.create_from_options(options)

#COOL ASS SPELLCASTING STUFF
SPELLS = {
    "Open_Palm":   {"name": "FIREBALL",      "word": "fireball"},
    "Closed_Fist": {"name": "CREATION",      "word": "creation"},
    "Pointing_Up": {"name": "TELEPORT",      "word": "teleport"},
    "ILoveYou":    {"name": "SUMMON UNDEAD", "word": "summon"},
}
POSE_WINDOW = 1.0   # seconds a pose stays valid
WORD_WINDOW = 1.5   # seconds a heard word stays valid
COOLDOWN = 1.5      # seconds between casts

pose_seen = {}
last_cast = -999.0

start = time.monotonic()
last_ts = -1
last_name = None

#main loop
running = True
last_debug = 0.0
while running:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        running = False
        break
    
    now = time.monotonic()
    ts = int((now - start) * 1000)
    if ts <= last_ts:
        ts = last_ts + 1
    last_ts = ts          # dedented so it always updates

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # recognize gestures
    result = recognizer.recognize_for_video(mp_image, ts)
    if result.gestures:
        top = result.gestures[0][0]
        hand = result.handedness[0][0].category_name
        label = f"{top.category_name} ({hand} hand, {top.score:.2f})"
        name = top.category_name
    else:
        label, name = "No hand", "No hand"

    if name in SPELLS:
        pose_seen[name] = now

    for pose, spell in SPELLS.items():
        pose_fresh = now - pose_seen.get(pose, -999.0) < POSE_WINDOW
        word_fresh = now - last_word_t < WORD_WINDOW
        if (pose_fresh and word_fresh
            and last_word == spell["word"]
            and now - last_cast > COOLDOWN):
            print(f"CASTING {spell['name']}")
            last_cast = now
            last_word = ""      # consume the word so it can't fire twice
            break
#debug print every 0.5s
    if now - last_debug > 0.5:
        last_debug = now
        spell = SPELLS.get(name)
        print(f"pose={name} word={last_word!r} "
            f"word_age={now - last_word_t:.1f}s "
            f"match={bool(spell) and last_word == spell['word']} "
            f"cooldown_ok={now - last_cast > COOLDOWN} "
            f"pose_fresh={now - pose_seen.get(name, -999.0) < POSE_WINDOW}")

#debug display
    cv2.putText(frame, f"pose: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"heard: {last_word}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    out.write(frame)
    cv2.imshow('Camera', frame)

    key = cv2.waitKey(1) & 0xFF            
    if key == ord('q'):
        running = False

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows() 