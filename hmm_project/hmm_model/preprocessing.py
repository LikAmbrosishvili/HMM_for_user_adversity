import numpy as np
import random
from typing import List, Dict

class ObservationPreprocessor:
    def __init__(self, alignment_threshold=35, velocity_tolerance=0.2):
        self.alignment_threshold = alignment_threshold
        self.velocity_tolerance = velocity_tolerance

    def categorize_click_frequency(self, clicks_per_minute):
        if clicks_per_minute <= 2:
            return 0  # Low
        elif 3 <= clicks_per_minute <= 6:
            return 1  # Medium
        else:
            return 2  # High

    def categorize_mouse_gaze_distance(self, mouse_x, mouse_y, gaze_x, gaze_y):
        distance = np.sqrt((mouse_x - gaze_x)**2 + (mouse_y - gaze_y)**2)
        if distance <= self.alignment_threshold:
            return 2  # high 
        elif self.alignment_threshold < distance <= 80:
            return 1  # Medium 
        else:
            return 0  # far

    def categorize_click_misfire(self, is_misfire):
        return 1 if is_misfire else 0

    def categorize_mouse_gaze_alignment(self, mouse_x, mouse_y, gaze_x, gaze_y):
        distance = np.sqrt((mouse_x - gaze_x)**2 + (mouse_y - gaze_y)**2)
        return 1 if distance <= self.alignment_threshold else 0

    def categorize_velocity_match(self, mouse_velocity, gaze_velocity):
        if mouse_velocity == 0 and gaze_velocity == 0:
            return 1
        diff_ratio = abs(mouse_velocity - gaze_velocity) / max(mouse_velocity, gaze_velocity)
        return 1 if diff_ratio <= self.velocity_tolerance else 0

    def categorize_cursor_reversal(self, direction_sequence):
        return 1 if len(set(direction_sequence)) > 2 else 0

    def process_observation(self, click_rate, is_misfire, mouse_x, mouse_y, gaze_x, gaze_y, mouse_velocity, gaze_velocity, direction_seq):
        return [
            self.categorize_click_frequency(click_rate),
            self.categorize_click_misfire(is_misfire),
            self.categorize_mouse_gaze_distance(mouse_x, mouse_y, gaze_x, gaze_y),
            self.categorize_velocity_match(mouse_velocity, gaze_velocity),
            self.categorize_cursor_reversal(direction_seq),
        ]

class ParticipantSimulator:
    def __init__(self, n_steps=100, random_seed=None):
        self.n_steps = n_steps
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        self.preprocessor = ObservationPreprocessor()
        print(f"Preprocessor created: {self.preprocessor}")


    def simulate(self) -> List[Dict]:
        raw_data = []
        mouse_x, mouse_y = 300, 300
        gaze_x, gaze_y = 305, 295

        directions = ['→', '←', '↑', '↓']

        for _ in range(self.n_steps):
            click_rate = np.random.choice([1, 2, 5, 8], p=[0.4, 0.4, 0.15, 0.05])
            is_misfire = np.random.rand() < 0.1
            mouse_dx, mouse_dy = np.random.randint(-5, 5), np.random.randint(-5, 5)
            gaze_dx, gaze_dy = np.random.randint(-5, 5), np.random.randint(-5, 5)

            mouse_x += mouse_dx
            mouse_y += mouse_dy
            gaze_x += gaze_dx
            gaze_y += gaze_dy

            mouse_velocity = np.random.uniform(0.5, 3.0)
            gaze_velocity = np.random.uniform(0.5, 3.0)

            dir_seq = random.choices(directions, k=5)

            raw_data.append({
                'click_rate': click_rate,
                'is_misfire': is_misfire,
                'mouse_x': mouse_x,
                'mouse_y': mouse_y,
                'gaze_x': gaze_x,
                'gaze_y': gaze_y,
                'mouse_velocity': mouse_velocity,
                'gaze_velocity': gaze_velocity,
                'direction_seq': dir_seq
            })

        return raw_data

    def generate_categorized_observations(self) -> np.ndarray:
        """ Process the raw simulated data into categorical observations """
        raw_data = self.simulate()
        processed = []

        for item in raw_data:
            observation = self.preprocessor.process_observation(
                click_rate=item['click_rate'],
                is_misfire=item['is_misfire'],
                mouse_x=item['mouse_x'],
                mouse_y=item['mouse_y'],
                gaze_x=item['gaze_x'],
                gaze_y=item['gaze_y'],
                mouse_velocity=item['mouse_velocity'],
                gaze_velocity=item['gaze_velocity'],
                direction_seq=item['direction_seq'],
            )
            processed.append(observation)

        return np.array(processed)
