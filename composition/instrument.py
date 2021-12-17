import os
import pickle

import ddsp.training
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
from IPython.display import Audio
from matplotlib.backends.backend_pdf import PdfPages

from composition.common import HOP_SIZE, SECOND, constant, generate_audio

COPPER = 4.236067978


def get_cents(midi):
    c = int(f'{midi:.2f}'.split('.')[1])
    if c >= 50:
        c -= 100
    return c


def get_note(midi):
    name = librosa.midi_to_note(midi)
    cents = get_cents(midi)

    if cents == 0:
        return name
    elif cents > 0:
        return f'{name} +{cents}'
    else:
        return f'{name} {cents}'


def note_formatter(x, pos=None):
    return get_note(x)


def pad(xs, duration, value=None):
    if value is None:
        value = xs.min()
    start = constant(duration, value)
    end = constant(duration * 2, value)

    return np.concatenate([start, xs, end])


def show(pitch, loudness, title=None, offset=0.0):
    steps = len(loudness)
    dur = steps / SECOND
    # px = 1 / plt.rcParams['figure.dpi']  # pixel in inches

    t = np.linspace(0, dur, steps) + offset

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(4 * COPPER, 4), gridspec_kw={'height_ratios': [COPPER, 1]})
    fig.subplots_adjust(hspace=0)

    if title:
        ax1.set_title(title)

    # pitch
    ax1.plot(t, pitch, color='blue')
    ax1.axes.xaxis.set_visible(False)
    ax1.yaxis.set_major_formatter(note_formatter)
    # ax1.axes.yaxis.set_visible(False)

    # loudness
    ax2.plot(t, loudness, color='blue')
    ax2.fill_between(t, min(loudness), loudness, color='red', alpha=0.75)
    ax2.axes.yaxis.set_visible(False)

    return fig


def phrase_from_audio(audio_path):
    audio, _ = librosa.load(audio_path, 16000)
    audio_features = ddsp.training.metrics.compute_audio_features(audio)

    phrase = Phrase(librosa.hz_to_midi(audio_features['f0_hz']), audio_features['loudness_db'])

    return phrase


class Phrase:
    def __init__(self, pitch, loudness):
        assert len(pitch) == len(loudness)

        self.pitch = pitch
        self.loudness = loudness

    def show(self):
        fig = show(self.pitch, self.loudness)
        plt.figure(fig)
        plt.show()

    def __len__(self):
        return len(self.pitch)


class Part:
    def __init__(self, part_name, instrument):
        self.part_name = part_name
        self.instrument = instrument
        self.phrases = []
        self.transpose = 0.

    def add_phrase(self, phrase):
        self.phrases.append(phrase)

    @property
    def pitch(self):
        return np.concatenate([p.pitch for p in self.phrases]) + self.transpose

    @property
    def loudness(self):
        return np.concatenate([p.loudness for p in self.phrases])

    def show(self, offset=None):
        if offset:
            fig = show(self.pitch[offset * SECOND:], self.loudness[offset * SECOND:], offset=offset,
                       title=self.part_name)
        else:
            fig = show(self.pitch, self.loudness, title=self.part_name)

        plt.figure(fig)
        plt.show()

    def audio(self):
        padded_pitch = pad(self.pitch, 2)
        padded_loudness = pad(self.loudness, 2, value=-110)
        return generate_audio(self.instrument, padded_pitch, padded_loudness)

    def play(self, offset=None):
        if offset:
            return Audio(self.audio()[offset * 16000:], rate=16000, normalize=False)

        return Audio(self.audio(), rate=16000, normalize=False)

    def __len__(self):
        return len(self.pitch)


class Score:
    def __init__(self, parts=None):
        if parts is None:
            self.parts = []
        else:
            self.parts = parts

        self.num_steps = max(len(p) for p in self.parts)
        self.duration = self.num_steps * HOP_SIZE

    def show(self):
        n = len(self.parts)
        fig, axes = plt.subplots(n, 1, figsize=(16, n * 4), sharex=True)

        for idx, part in enumerate(self.parts):
            steps = len(part)
            dur = steps / SECOND
            t = np.linspace(0, dur, steps)

            ax2 = axes[idx].twinx()

            axes[idx].plot(t, part.loudness, color='red')
            ax2.plot(t, part.pitch, color='blue')

            plt.title(part.part_name)

    def audio(self):
        result = np.zeros(self.duration)
        for p in self.parts:
            audio = p.audio()
            result[:len(audio)] += audio

        return result

    def play(self):
        return Audio(self.audio(), rate=16000, normalize=False)

    def save(self, name, base_path='./audio-data/original'):
        path = os.path.join(base_path, name)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'score.pkl'), 'wb') as f:
            pickle.dump(self, f)
        for part in self.parts:
            soundfile.write(os.path.join(path, f'{name}-{part.part_name}.wav'), part.audio(), 16000)

        pp = PdfPages(os.path.join(path, f'{name}.pdf'))
        self.show()
        pp.savefig()
        pp.close()
