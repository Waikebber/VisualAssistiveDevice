from collections import defaultdict
import threading

class EventBus:
    def __init__(self):
        # Dictionary of event names to subscriber functions
        self._subscribers = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, event_name, callback):
        """Subscribe a callback function to an event."""
        with self._lock:
            self._subscribers[event_name].append(callback)

    def unsubscribe(self, event_name, callback):
        """Unsubscribe a callback function from an event."""
        with self._lock:
            if callback in self._subscribers[event_name]:
                self._subscribers[event_name].remove(callback)

    def publish(self, event_name, *args, **kwargs):
        """Publish an event to all subscribers."""
        with self._lock:
            for callback in self._subscribers[event_name]:
                # Run each callback in a separate thread for async execution
                threading.Thread(target=callback, args=args, kwargs=kwargs).start()
