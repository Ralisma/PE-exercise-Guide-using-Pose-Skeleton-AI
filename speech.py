import pyttsx3
import threading

class SpeechManager:
    def __init__(self, rate=165):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.speech_lock = threading.Lock()

    def speak_sync(self, text):
        print("[SAY]", text)
        with self.speech_lock:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"TTS failed: {str(e)}")

    def speak_async(self, text):
        def _speak():
            self.speak_sync(text)
        threading.Thread(target=_speak, daemon=True).start()
