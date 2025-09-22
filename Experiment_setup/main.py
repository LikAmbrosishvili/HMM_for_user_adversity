"""
main.py

This is the entry point for the user interaction logging experiment.

It integrates:
- EventDetector: for detecting keyboard and mouse activity
- SessionLogger (session_handling): for logging structured behavior to a CSV file (1 row per second)
- Progress indicator: prints a subtle dot every second to show the experiment is running
exit

The experiment stops gracefully when the user types "exit".
"""

from event_detector import EventDetector
from session_handling import SessionLogger
import threading
import time

def main():
    #initialize the session logger
    logger = SessionLogger()
    logger.start()

    #handle incoming events from the EventDetector
    def handle_event(event_type, data):
        logger.add_event(event_type, data)

    # stop the session cleanly when "exit" is typed
    def stop_experiment():
        print("\n Stopping experiment...")
        logger.stop()

    # Print progress 
    def show_progress():
        print(" Experiment running...\n", flush=True)
        while not logger.stop_flag:
            print(".", end="", flush=True)
            time.sleep(1)

   
    threading.Thread(target=show_progress, daemon=True).start()
    # Start event detection (blocks until 'exit' is typed)
    detector = EventDetector(handle_event, stop_experiment)
    detector.start()

if __name__ == "__main__":
    main()


