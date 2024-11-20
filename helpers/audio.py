import multiprocessing
from speakers.generative_audio import speak

class AudioHandler:
    def __init__(self):
        self.audio_process = None
    
    def speak_async(self, text):
        """Speak text asynchronously using multiprocessing."""
        if self.audio_process is None or not self.audio_process.is_alive():
            self.audio_process = multiprocessing.Process(
                target=speak,
                args=(text, 3, 90)
            )
            self.audio_process.start()