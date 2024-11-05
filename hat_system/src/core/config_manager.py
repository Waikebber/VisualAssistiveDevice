import json
import os

class ConfigManager:
    def __init__(self, config_path="/data/stereo/cam_config.json"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """Load the configuration settings from a JSON file."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as file:
                return json.load(file)
        else:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

    def get_config(self, key, default=None):
        """Get a configuration setting by key, with an optional default."""
        return self.config.get(key, default)
