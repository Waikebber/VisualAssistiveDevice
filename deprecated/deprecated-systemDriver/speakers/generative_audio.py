from gtts import gTTS
from io import BytesIO
import pygame
from enum import Enum
import subprocess
from typing import Union

APPROPRIATE_VOL: int = 30
WAITING_TIME: int = 10  # for playing a single audio to end

class SpeakerChannel(Enum):
    LEFT: int = 1
    RIGHT: int = 2
    BOTH: int = 3

def speak(text, channel: Union[int, SpeakerChannel] = SpeakerChannel.BOTH, vol: int = APPROPRIATE_VOL):
    """Speak text through specified audio channel (LEFT, RIGHT, or BOTH).
    :param text: str of full content to say
    :param channel: int/SpeakerChannel (1: left, 2: right, 3: both), defaults to both
    :param vol: int volume (as a percent out of 100), default is 30%
    """

    mp3_fp = BytesIO()
    tts = gTTS(text=text, lang='en')
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    channel = SpeakerChannel(channel) if isinstance(channel, int) else channel

    if channel == SpeakerChannel.BOTH:
        set_volume(vol, vol)
    elif channel == SpeakerChannel.LEFT:
        set_volume(vol, 0)
    elif channel == SpeakerChannel.RIGHT:
        set_volume(0, vol)

    pygame.mixer.init()
    pygame.mixer.music.load(mp3_fp)
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(WAITING_TIME)

def set_volume(left_volume, right_volume):
    """Set the volume for left and right speakers."""
    command = [
        "amixer",
        "-c", "1",
        "sset",
        "Speaker",
        f"{left_volume}%,{right_volume}%"
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Volume set to: Left {left_volume}%, Right {right_volume}%")
    except subprocess.CalledProcessError as e:
        print(f"Error setting volume: {e}")
        print(f"Error output: {e.stderr}")
