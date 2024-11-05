from core.event_bus import EventBus

class SoundManager:
    def __init__(self, event_bus: EventBus):
        """Initialize the SoundManager with an event bus."""
        self.event_bus = event_bus

        # Subscribe to relevant events for audio feedback
        self.event_bus.subscribe("threshold_distance_crossed", self.alert_threshold_crossed)
        self.event_bus.subscribe("object_recognition_completed", self.announce_recognition_results)

    def play_audio_feedback(self, message):
        """Simulates audio playback by printing a message. Replace this with actual audio playback logic."""
        print(f"Audio Feedback: {message}")

    def alert_threshold_crossed(self, distance, object_name=None):
        """Handles the event when an object crosses the threshold distance."""
        if object_name:
            feedback_text = f"Alert: {object_name} detected within {distance:.2f} meters."
        else:
            feedback_text = f"Alert: Object detected within {distance:.2f} meters."
        self.play_audio_feedback(feedback_text)

    def announce_recognition_results(self, results_with_distance):
        """Handles the event after image recognition, announcing object names and distances."""
        if not results_with_distance:
            self.play_audio_feedback("No recognizable objects within range.")
            return
        
        # Construct feedback message for each recognized object
        message = "Detected objects: "
        for obj_name, distance, confidence in results_with_distance:
            if distance is not None:
                message += f"{obj_name} at {distance:.2f} meters. "
            else:
                message += f"{obj_name} detected. "
        
        self.play_audio_feedback(message)
