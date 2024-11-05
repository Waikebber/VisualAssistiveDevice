class SystemState:
    def __init__(self):
        self.state = {
            "stereovision_active": False,
            "audio_playback_active": False,
            "distance_threshold_crossed": False,
            "button_pressed": False
        }

    def set_state(self, key, value):
        """Set the state of a given key."""
        if key in self.state:
            self.state[key] = value
        else:
            raise KeyError(f"Invalid state key: {key}")

    def get_state(self, key):
        """Get the state of a given key."""
        if key in self.state:
            return self.state[key]
        else:
            raise KeyError(f"Invalid state key: {key}")
