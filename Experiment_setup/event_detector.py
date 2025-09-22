# event_detector.py

from pynput import mouse, keyboard
import threading
import time

class EventDetector:
    """
event_detector.py

This module defines the EventDetector class, which captures real-time user interactions
such as mouse clicks, mouse movements (hover), keyboard key presses, and backspace events
using the pynput library. It is designed for use in human-computer interaction experiments.

Key Features:
- Tracks mouse click coordinates and hover positions
- Detects key presses and backspace
- Monitors for an "exit" sequence (typing 'exit') to gracefully stop the session
- Sends structured events to a user-defined callback function for logging or processing

Intended for integration into experimental pipelines that log user behavior, e.g., in VS Code.
"""

    def __init__(self, event_callback, exit_callback):
        """
        Parameters:
        - event_callback: function(event_type: str, data: dict)
        - exit_callback: function() to stop the experiment
        """
        self.event_callback = event_callback
        self.exit_callback = exit_callback
        self.exit_buffer = []
        self.keyboard_listener = None
        self.mouse_listener = None

    def start(self):
        self.mouse_listener = mouse.Listener(
            on_click=self.on_click,
            on_move=self.on_move
        )
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_press
        )

        self.mouse_listener.start()
        self.keyboard_listener.start()

        # listeners run in the background
        self.keyboard_listener.join()
        self.mouse_listener.join()

    def on_click(self, x, y, button, pressed):
        if pressed:
            self.event_callback("click", {"x": x, "y": y, "button": str(button)})

    def on_move(self, x, y):
        self.event_callback("hover", {"x": x, "y": y})

    def on_press(self, key):
        try:
            key_str = key.char.lower()
        except AttributeError:
            key_str = None

        if key == keyboard.Key.backspace:
            self.event_callback("backspace", {})

        if key_str:
            self.event_callback("keypress", {"key": key_str})
            self.exit_buffer.append(key_str)
            if len(self.exit_buffer) > 4:
                self.exit_buffer.pop(0)
            if ''.join(self.exit_buffer) == 'exit':
                print(" Exit command typed.")
                self.exit_callback()
#