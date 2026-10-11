import os, base64, time, asyncio, threading, cv2, re
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import sounddevice as sd
from elevenlabs import (
    AudioFormat,
    CommitStrategy,
    ElevenLabs,
    RealtimeAudioOptions,
    RealtimeEvents,
)

from dotenv import load_dotenv
load_dotenv() 

DEBUG = False
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

MIN_SCORE = 0.6
HOLD_TIME = 0.25
SAME_WORD_LOCKOUT = 2.5

pose_since = {}          # pose -> when it became continuously visible
pose_ready = {}          # pose -> last time it was held steadily
last_cast_word = ""

pose_seen = {}
last_cast = -999.0
last_ts = -1

def cast(spell):
    print(f"CASTING {spell['name']}!")

#TRANSCRIPTION STUFF
last_word = ""
last_word_t = 0.0
heard = ("", 0.0)  # (word, time)

SAMPLE_RATE = 48000
BLOCKSIZE = 4800  # 100 ms per block

def clean(text):
    words = re.findall(r"[a-z]+", text.lower())
    return words[-1] if words else ""


async def transcribe_forever():
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
        global heard
        raw = data.get("text", "")
        with open("stt_log.txt", "a") as f:           # raw log for debugging
            f.write(f"{time.time():.2f} {raw!r}\n")
        word = clean(raw)
        if word:
            heard = (word, time.monotonic())
    
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
 
start = time.monotonic()


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
    min_hand_detection_confidence=MIN_SCORE,
    min_tracking_confidence=MIN_SCORE,
)
recognizer = vision.GestureRecognizer.create_from_options(options)


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
        name = top.category_name
    else:
        name = "No hand"

    word, word_t = heard                       # one snapshot per frame
    score = top.score if result.gestures else 0.0

    current = name if (name in SPELLS and score >= MIN_SCORE) else None
    for p in list(pose_since):
        if p != current:
            del pose_since[p]
    if current:
        pose_since.setdefault(current, now)
        if now - pose_since[current] >= HOLD_TIME:
            pose_ready[current] = now

    word_fresh = now - word_t < WORD_WINDOW
    for pose, spell in SPELLS.items():
        if (now - pose_ready.get(pose, -999.0) < POSE_WINDOW
                and word_fresh
                and word == spell["word"]
                and now - last_cast > COOLDOWN
                and not (word == last_cast_word
                         and now - last_cast < SAME_WORD_LOCKOUT)):
            cast(spell)
            last_cast, last_cast_word = now, word
            heard = ("", 0.0)
            break
#debug stuff
    if DEBUG:
        cv2.flip(frame, 1)  # mirror image
        cv2.putText(frame, f"pose: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"heard: {last_word}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        out.write(frame)
        cv2.imshow('Camera', frame)
        if now - last_debug > 0.5:
            last_debug = now
            spell = SPELLS.get(name)
            print(f"pose={name} word={last_word!r} "
                f"word_age={now - last_word_t:.1f}s "
                f"match={bool(spell) and last_word == spell['word']} "
                f"cooldown_ok={now - last_cast > COOLDOWN} "
                f"pose_fresh={now - pose_seen.get(name, -999.0) < POSE_WINDOW}")
    key = cv2.waitKey(1) & 0xFF 
    if key == ord('q'):
        running = False

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows() 