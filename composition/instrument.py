import os
import pickle

import ddsp.training
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
from IPython.display import Audio
from matplotlib import animation, gridspec

from composition.common import HOP_SIZE, SECOND, constant, generate_audio

# from matplotlib.backends.backend_pdf import PdfPages

COPPER = 4.236067978
FPS = 30


def get_cents(midi):
    c = int(f'{midi:.2f}'.split('.')[1])
    if c > 50:
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


v_get_note = np.vectorize(get_note)


def note_formatter(x, _pos=None):
    return get_note(x)


def pad(xs, duration, value=None):
    if value is None:
        value = xs.min()
    start = constant(duration, value)
    end = constant(duration * 4, value)

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
    y_start = np.floor(pitch.min())
    y_end = np.ceil(pitch.max())
    pitch_nums = np.arange(y_start, y_end + 1)
    ax1.set_yticks(pitch_nums)

    # loudness
    ax2.plot(t, loudness, color='blue')
    ax2.fill_between(t, min(loudness), loudness, color='red', alpha=0.75)
    ax2.axes.yaxis.set_visible(False)
    x_ticks = np.concatenate([t, t[-1:] + 1 / SECOND])
    ax2.set_xticks(x_ticks[::SECOND])

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

    def audio(self, instrument):
        padded_pitch = pad(self.pitch, 2)
        padded_loudness = pad(self.loudness, 2, value=-110)
        return generate_audio(instrument, padded_pitch, padded_loudness)

    def play(self, instrument):
        return Audio(self.audio(instrument), rate=16000, normalize=False)

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
            fig = show(self.pitch[int(offset * SECOND):], self.loudness[int(offset * SECOND):], offset=offset,
                       title=self.part_name)
        else:
            fig = show(self.pitch, self.loudness, title=self.part_name)

        plt.figure(fig)
        plt.show()

    def audio(self):
        padded_pitch = pad(self.pitch, 2)
        padded_loudness = pad(self.loudness, 2, value=-110)
        return generate_audio(self.instrument, padded_pitch, padded_loudness)

    def video(self, path, codec='h264'):
        fig = show(self.pitch, self.loudness, self.part_name)
        vline1 = fig.get_axes()[0].axvline(0.)
        vline2 = fig.get_axes()[1].axvline(0.)

        def show_time(t):
            vline1.set_data([t / FPS, t / FPS], [0, 1])
            vline2.set_data([t / FPS, t / FPS], [0, 1])

        n_frames = int(FPS * (len(self.pitch) / SECOND))
        anim = animation.FuncAnimation(fig, show_time, frames=n_frames)
        anim.save(path, writer=animation.FFMpegWriter(FPS, codec))

    def play(self, offset=None):
        if offset:
            return Audio(self.audio()[int(offset * 16000):], rate=16000, normalize=False)

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

    def show(self, begin=0, end=-1):
        if end > 0:
            steps = end - begin
        else:
            steps = max((len(p) for p in self.parts))

        parts = [p for p in self.parts if any(p.loudness[begin:end] > -120)]
        start = begin / SECOND
        stop = end / SECOND
        t = np.linspace(start, stop, steps)

        n = len(parts)
        figure = plt.figure(figsize=(1920 / 100, 1080 / 100))
        plt.gcf().set_dpi(100)
        outer_grid = gridspec.GridSpec(n, 1)

        for part, cell in zip(parts, outer_grid):
            inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, cell, hspace=0, height_ratios=(COPPER, 1))

            # From here we can plot using inner_grid's SubplotSpecs
            ax1 = plt.subplot(inner_grid[0, 0])
            ax2 = plt.subplot(inner_grid[1, 0])

            ax1.set_title(part.part_name)

            # pitch
            ax1.plot(t, part.pitch[begin:end], color='blue')
            ax1.axes.xaxis.set_visible(False)
            ax1.yaxis.set_major_formatter(note_formatter)
            y_start = np.floor(part.pitch[begin:end].min())
            y_end = np.ceil(part.pitch[begin:end].max())
            pitch_nums = np.arange(y_start, y_end + 1)
            ax1.set_yticks(pitch_nums)

            # loudness
            ax2.plot(t, part.loudness[begin:end], color='blue')
            ax2.fill_between(t, min(part.loudness[begin:end]), part.loudness[begin:end], color='red', alpha=0.75)
            ax2.axes.yaxis.set_visible(False)
            x_ticks = np.concatenate([t, t[-1:] + 1 / SECOND])
            ax2.set_xticks(x_ticks[::SECOND])

        return figure

    def video(self, path, begin=0, end=-1, codec='h264'):
        fig = self.show(begin, end)
        start = begin / SECOND
        vlines = [ax.axvline(start) for ax in fig.get_axes()]

        def show_time(t):
            for vl in vlines:
                vl.set_data([start + t / FPS, start + t / FPS], [0, 1])

        if end > 0:
            steps = end - begin
        else:
            steps = max((len(p) for p in self.parts))
        n_frames = int(FPS * (steps / SECOND))
        anim = animation.FuncAnimation(fig, show_time, frames=n_frames)
        anim.save(path, writer=animation.FFMpegWriter(FPS, codec), dpi=100)

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

        # pp = PdfPages(os.path.join(path, f'{name}.pdf'))
        # self.show()
        # pp.savefig()
        # pp.close()
