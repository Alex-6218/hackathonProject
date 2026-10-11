
import asyncio
import base64
import json
import os
import queue
import re
import socket
import threading
import time
from typing import Callable, Dict, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sounddevice as sd
from dotenv import load_dotenv
from elevenlabs import (
    AudioFormat,
    CommitStrategy,
    ElevenLabs,
    RealtimeAudioOptions,
    RealtimeEvents,
)

load_dotenv()


class GodotBridge:
    """Fire-and-forget JSON-over-UDP sender for talking to a local Godot game.

    Host and port default to the GODOT_HOST / GODOT_PORT environment variables
    (or 127.0.0.1:4242), so both sides can be configured without code changes.
    """

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        self.addr = (
            host or os.getenv("GODOT_HOST", "127.0.0.1"),
            int(port or os.getenv("GODOT_PORT", "4242")),
        )
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

    def send(self, message: Dict) -> None:
        """Send one JSON message. Never raises: a missing game must not stop recognition."""
        try:
            self.sock.sendto(json.dumps(message).encode("utf-8"), self.addr)
        except OSError as exc:
            print(f"[godot] send failed: {exc}")

    def send_spell(self, spell_id: str, spell: Dict[str, str]) -> None:
        self.send(
            {
                "type": "spell_cast",
                "id": spell_id,
                "spell": spell["name"],
                "word": spell["word"],
                "t": time.time(),
            }
        )

    def close(self) -> None:
        self.sock.close()


class SpellRecognizer:
    # Gesture label -> spell metadata. The gesture label also acts as a spell ID.
    SPELLS = {
        "Open_Palm": {"name": "FIREBALL", "word": "fireball"},
        "Pointing_Up": {"name": "TELEPORT", "word": "teleport"},
        "Closed_Fist": {"name": "CONSTRUCT", "word": "construct"},
        "ILoveYou": {"name": "SUMMON UNDEAD", "word": "undead"},
    }

    # Recognition / casting settings (seconds unless otherwise noted).
    POSE_WINDOW = 1.0
    WORD_WINDOW = 1.5
    COOLDOWN = 0.8
    MIN_SCORE = 0.5
    HOLD_TIME = 0.25
    SAME_WORD_LOCKOUT = 2.5

    SAMPLE_RATE = 48_000
    BLOCKSIZE = 4_800  # 100 ms of audio per block

    def __init__(
        self,
        camera_index: int = 2,
        microphone_device: int = 4,
        model_asset_path: str = "gesture_recognizer.task",
        debug: bool = False,
        record_video: bool = False,
        on_spell_cast: Optional[Callable[[str, Dict[str, str]], None]] = None,
        bridge: Optional[GodotBridge] = None,
    ):
        """Configure the recognizer without starting the camera or microphone.

        bridge, if supplied, receives every confirmed spell (sent to Godot).
        on_spell_cast, if supplied, is also called as on_spell_cast(spell_id, spell).
        """
        self.bridge = bridge
        self.camera_index = camera_index
        self.microphone_device = microphone_device
        self.model_asset_path = model_asset_path
        self.debug = debug
        self.record_video = record_video
        self.on_spell_cast = on_spell_cast

        # Gesture and spell state (previously module-level variables).
        self.pose_since = {}       # gesture -> time continuous hold began
        self.pose_ready = {}       # gesture -> most recent time it met HOLD_TIME
        self.last_cast = -999.0
        self.last_cast_word = ""
        self.last_ts = -1
        self.start_time = None

        # Transcript callbacks run outside the camera-processing flow. A queue
        # avoids having the camera loop directly share mutable transcript state
        # with the speech-recognition thread.
        self.transcript_queue = queue.Queue()
        self.heard = ("", 0.0)     # latest usable (word, monotonic timestamp)
        self.last_transcript = ""

        # Managed resources and lifecycle state.
        self.cam = None
        self.out = None
        self.recognizer = None
        self.running = False
        self.stop_event = threading.Event()
        self.transcriber_thread = None
        self.last_debug = 0.0

    def initialize_camera(self):
        """Open the camera and optionally initialize the debug video writer."""
        self.cam = cv2.VideoCapture(self.camera_index)
        if not self.cam.isOpened():
            self.cam.release()
            self.cam = None
            raise RuntimeError(f"Could not open camera device {self.camera_index}")

        if self.record_video:
            width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out = cv2.VideoWriter(
                "output.mp4", fourcc, 20.0, (width, height)
            )
            if not self.out.isOpened():
                self.out.release()
                self.out = None
                raise RuntimeError("Could not initialize output.mp4 writer")

    def initialize_gesture_recognizer(self):
        """Load and configure the MediaPipe gesture recognizer."""
        options = vision.GestureRecognizerOptions(
            base_options=python.BaseOptions(
                model_asset_path=self.model_asset_path
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=self.MIN_SCORE,
            min_tracking_confidence=self.MIN_SCORE,
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    @staticmethod
    def clean_transcript(text: str) -> str:
        """Return the last alphabetic word from a transcript, as in the original."""
        words = re.findall(r"[a-z]+", text.lower())
        return words[-1] if words else ""

    def detect_gesture(self, frame, now: float) -> Tuple[str, Optional[str], float]:
        """Analyze one frame; return (gesture name, handedness, confidence)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # MediaPipe VIDEO mode requires monotonically increasing timestamps.
        elapsed = 0.0 if self.start_time is None else now - self.start_time
        timestamp_ms = int(elapsed * 1000)
        if timestamp_ms <= self.last_ts:
            timestamp_ms = self.last_ts + 1
        self.last_ts = timestamp_ms

        result = self.recognizer.recognize_for_video(mp_image, timestamp_ms)
        if result.gestures and result.gestures[0]:
            top = result.gestures[0][0]
            hand = None
            if result.handedness and result.handedness[0]:
                hand = result.handedness[0][0].category_name
            return top.category_name, hand, float(top.score)

        return "No hand", None, 0.0

    def update_pose_state(self, current_pose: Optional[str], now: float):
        """Track continuous holds and remember when a pose becomes ready."""
        for pose in list(self.pose_since):
            if pose != current_pose:
                del self.pose_since[pose]

        if current_pose:
            self.pose_since.setdefault(current_pose, now)
            if now - self.pose_since[current_pose] >= self.HOLD_TIME:
                self.pose_ready[current_pose] = now

    async def transcribe_forever(self):
        """Stream microphone audio to ElevenLabs until stop is requested."""
        loop = asyncio.get_running_loop()
        audio_queue = asyncio.Queue()
        session_ready = asyncio.Event()

        client = ElevenLabs(api_key=os.getenv("API_KEY"))
        connection = await client.speech_to_text.realtime.connect(
            RealtimeAudioOptions(
                model_id="scribe_v2_realtime",
                audio_format=AudioFormat.PCM_48000,
                sample_rate=self.SAMPLE_RATE,
                commit_strategy=CommitStrategy.VAD,
            )
        )

        def on_session_started(data):
            print("[stt] session started")
            # The SDK callback may run in a different thread from this loop.
            loop.call_soon_threadsafe(session_ready.set)

        def on_transcript(data):
            raw = data.get("text", "")
            try:
                with open("stt_log.txt", "a", encoding="utf-8") as log_file:
                    log_file.write(f"{time.time():.2f} {raw!r}\n")
            except OSError as exc:
                print(f"[stt log warning] {exc}")

            word = self.clean_transcript(raw)
            if word:
                self.transcript_queue.put((word, time.monotonic(), raw))

        def on_error(error):
            print(f"[stt error] {error}")

        def on_close():
            print("[stt] connection closed")

        connection.on(RealtimeEvents.SESSION_STARTED, on_session_started)
        connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, on_transcript)
        connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, on_transcript)
        connection.on(RealtimeEvents.ERROR, on_error)
        connection.on(RealtimeEvents.CLOSE, on_close)

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"[audio warning] {status}")
            loop.call_soon_threadsafe(audio_queue.put_nowait, bytes(indata))

        # Keep the audio stream local to the coroutine/session that owns it.
        with sd.RawInputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=self.BLOCKSIZE,
            device=self.microphone_device,
            callback=audio_callback,
        ):
            # Poll while waiting for startup so shutdown is noticed promptly;
            # fail the session if the service never reports readiness.
            startup_deadline = loop.time() + 15.0
            while (
                not self.stop_event.is_set()
                and not session_ready.is_set()
                and loop.time() < startup_deadline
            ):
                await asyncio.sleep(0.1)

            if not self.stop_event.is_set() and not session_ready.is_set():
                raise TimeoutError("Speech-recognition session did not start")

            while not self.stop_event.is_set():
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue

                await connection.send(
                    {
                        "audio_base_64": base64.b64encode(chunk).decode("ascii"),
                        "sample_rate": self.SAMPLE_RATE,
                    }
                )

        # Check the installed ElevenLabs SDK version for its supported explicit
        # connection/client close method if you need immediate network teardown.

    def start_transcriber(self):
        """Start one background worker, with retry behavior from the original."""
        if self.transcriber_thread and self.transcriber_thread.is_alive():
            return

        def runner():
            while not self.stop_event.is_set():
                try:
                    asyncio.run(self.transcribe_forever())
                    if not self.stop_event.is_set():
                        print("[stt] session ended; retrying in 2 seconds")
                except Exception as exc:
                    if not self.stop_event.is_set():
                        print(f"[stt] crashed: {exc}; retrying in 2 seconds")
                if not self.stop_event.is_set():
                    self.stop_event.wait(2.0)

        self.transcriber_thread = threading.Thread(
            target=runner, name="SpellRecognizerTranscriber", daemon=True
        )
        self.transcriber_thread.start()

    def _update_latest_transcript(self):
        """Move queued speech events into the latest-transcript state."""
        while True:
            try:
                word, timestamp, raw = self.transcript_queue.get_nowait()
            except queue.Empty:
                break
            self.heard = (word, timestamp)
            self.last_transcript = raw

    def check_spell(self, current_time: float) -> Optional[Tuple[str, Dict[str, str]]]:
        """Return (spell_id, spell data) when a valid cast is detected."""
        word, word_time = self.heard
        if not word or current_time - word_time >= self.WORD_WINDOW:
            return None

        if current_time - self.last_cast <= self.COOLDOWN:
            return None

        if (
            word == self.last_cast_word
            and current_time - self.last_cast < self.SAME_WORD_LOCKOUT
        ):
            return None

        for pose, spell in self.SPELLS.items():
            pose_age = current_time - self.pose_ready.get(pose, -999.0)
            if pose_age < self.POSE_WINDOW and word == spell["word"]:
                return pose, spell

        return None

    def cast_spell(self, spell_id: str, spell: Dict[str, str], now: float):
        """Record a successful cast and notify the host/integration layer."""
        word = spell["word"]
        self.last_cast = now
        self.last_cast_word = word
        self.heard = ("", 0.0)  # Consume this keyword so it cannot cast again.

        print(f"CASTING {spell['name']}!")

        # Notify Godot (game-side effects live there, not in this class).
        if self.bridge is not None:
            self.bridge.send_spell(spell_id, spell)

        if self.on_spell_cast is not None:
            self.on_spell_cast(spell_id, spell)

    def update_debug_display(self, frame, gesture: str, score: float, now: float):
        """Draw optional debug overlays and write video only when requested."""
        word, word_time = self.heard
        if self.debug:
            cv2.putText(
                frame, f"pose: {gesture} ({score:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
            cv2.putText(
                frame, f"heard: {word}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )
            cv2.imshow("Camera", frame)

            if now - self.last_debug > 0.5:
                self.last_debug = now
                matching_spell = self.SPELLS.get(gesture)
                matches = bool(matching_spell) and word == matching_spell["word"]
                print(
                    f"pose={gesture} word={word!r} "
                    f"word_age={now - word_time:.1f}s "
                    f"match={matches} "
                    f"cooldown_ok={now - self.last_cast > self.COOLDOWN}"
                )

        if self.out is not None:
            self.out.write(frame)

    def process_frame(self):
        """Read and process one camera frame; return False if capture failed."""
        if self.cam is None or self.recognizer is None:
            raise RuntimeError("Camera and gesture recognizer must be initialized first")

        ret, frame = self.cam.read()
        if not ret:
            print("Failed to grab frame")
            return False

        now = time.monotonic()
        gesture, _hand, score = self.detect_gesture(frame, now)
        current_pose = (
            gesture
            if gesture in self.SPELLS and score >= self.MIN_SCORE
            else None
        )

        self.update_pose_state(current_pose, now)
        self._update_latest_transcript()

        spell_match = self.check_spell(now)
        if spell_match is not None:
            spell_id, spell = spell_match
            self.cast_spell(spell_id, spell, now)

        self.update_debug_display(frame, gesture, score, now)
        return True

    def run(self):
        """Initialize resources and run the recognition loop until stopped."""
        self.running = True
        try:
            self.initialize_camera()
            self.initialize_gesture_recognizer()
            self.start_time = time.monotonic()
            self.start_transcriber()

            while self.running and not self.stop_event.is_set():
                if not self.process_frame():
                    break
                if self.debug and (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
        finally:
            self.stop()

    def stop(self):
        """Request shutdown and release resources that have been initialized."""
        self.running = False
        self.stop_event.set()

        if self.cam is not None:
            self.cam.release()
            self.cam = None

        if self.out is not None:
            self.out.release()
            self.out = None

        if self.recognizer is not None:
            self.recognizer.close()
            self.recognizer = None

        if self.debug:
            cv2.destroyAllWindows()

        if (
            self.transcriber_thread is not None
            and self.transcriber_thread.is_alive()
            and threading.current_thread() is not self.transcriber_thread
        ):
            self.transcriber_thread.join(timeout=2.0)

        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None


if __name__ == "__main__":
    SpellRecognizer(
        debug=True,
        record_video=True,
        bridge=GodotBridge(),
    ).run()
