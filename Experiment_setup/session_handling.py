"""
session_handling.py

Handles logging user interaction data into a CSV file. Each session is automatically
saved in the 'sessions/' folder with an incremented filename (e.g., user_actions_1.csv).
One row per second is written, capturing timestamp, mouse/keyboard events, and idle time.
"""

import os
import csv
import re
import threading
import time

class SessionLogger:
    def __init__(self, base_filename="user_actions"):
        self.fieldnames = ['timestamp', 'mouse_click_x', 'mouse_click_y',
                           'mouse_hover', 'is_idle', 'key_press', 'backspace']
        self.buffer = {}
        self.buffer_lock = threading.Lock()
        self.stop_flag = False
        self.last_logged_time = int(time.time()) - 1

        self.base_filename = base_filename
        self.log_dir = self._get_sessions_folder()
        self.filename = self._generate_filename()
        self._init_file()
        self.writer_thread = threading.Thread(target=self._write_loop, daemon=True)

    def _get_sessions_folder(self):
        # Get absolute path to "sessions" folder relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sessions_dir = os.path.join(base_dir, "sessions")
        os.makedirs(sessions_dir, exist_ok=True)
        return sessions_dir

    def _generate_filename(self):
        # Ensure each session file is saved with an incremented name
        files = os.listdir(self.log_dir)
        pattern = re.compile(f"{re.escape(self.base_filename)}_(\\d+)\\.csv")
        nums = [int(m.group(1)) for f in files if (m := pattern.match(f))]
        next_num = max(nums) + 1 if nums else 1
        return os.path.join(self.log_dir, f"{self.base_filename}_{next_num}.csv")

    def _init_file(self):
        with open(self.filename, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def start(self):
        self.writer_thread.start()

    def stop(self):
        self.stop_flag = True
        self.writer_thread.join()
        print(f"\n📁 Session saved to: {self.filename}")

    def add_event(self, event_type, data):
        timestamp = int(time.time())
        with self.buffer_lock:
            if timestamp not in self.buffer:
                self.buffer[timestamp] = {field: "" for field in self.fieldnames}
                self.buffer[timestamp]['timestamp'] = timestamp

            row = self.buffer[timestamp]

            if event_type == "click":
                row['mouse_click_x'] = data.get("x")
                row['mouse_click_y'] = data.get("y")
            elif event_type == "hover":
                row['mouse_hover'] = f"({data.get('x')},{data.get('y')})"
            elif event_type == "keypress":
                row['key_press'] = data.get("key")
            elif event_type == "backspace":
                row['backspace'] = "Backspace"

    def _write_loop(self):
        while not self.stop_flag:
            time.sleep(1)
            current_time = int(time.time())
            with self.buffer_lock:
                for t in range(self.last_logged_time + 1, current_time + 1):
                    row = self.buffer.get(t, {field: "" for field in self.fieldnames})
                    row['timestamp'] = t
                    if row == {field: "" for field in self.fieldnames}:
                        row['timestamp'] = t
                        row['is_idle'] = "idle"
                    with open(self.filename, mode='a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                        writer.writerow(row)
                    if t in self.buffer:
                        del self.buffer[t]
                self.last_logged_time = current_time
