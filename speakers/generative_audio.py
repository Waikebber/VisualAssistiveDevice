from espeakng import ESpeakNG
from enum import Enum
import subprocess
from typing import Union

APPROPRIATE_VOL: int = 30
DEFAULT_SPEED: int = 130
DEFAULT_PITCH: int = 40

esng = ESpeakNG()
esng.voice = 'en-us+f3'
esng.speed = DEFAULT_SPEED
esng.pitch = DEFAULT_PITCH

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

    channel = SpeakerChannel(channel) if isinstance(channel, int) else channel

    if channel == SpeakerChannel.BOTH:
        set_volume(vol, vol)
    elif channel == SpeakerChannel.LEFT:
        set_volume(vol, 0)
    elif channel == SpeakerChannel.RIGHT:
        set_volume(0, vol)

    esng.say(text)

def set_volume(left_volume, right_volume):
    """Set the volume for left and right speakers."""
    command = [
        "amixer",
        "-c", "2",
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
