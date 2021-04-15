import os
import pickle

import numpy as np


class Instrument:
    def __init__(self, name: str, pitch: np.array, loudness: np.array, db_shift: float = 0.):
        super().__init__()

        assert len(pitch) == len(loudness)

        self.name = name
        self.pitch = pitch
        self.loudness = loudness
        self.db_shift = db_shift
        self.model_dir = f'./pretrained/{self.name.lower()}'

        with open(os.path.join(self.model_dir, 'dataset_statistics.pkl'), 'rb') as f:
            stats = pickle.load(f)
        self.quantile_transform = stats['quantile_transform']

    def get_audio(self):
        pass

    def plot_score(self):
        pass
