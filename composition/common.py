import os
import pickle
from collections import deque
from itertools import accumulate, cycle

import ddsp.training
import gin
import librosa
import numpy as np

from utils import fit_quantile_transform

SAMPLE_RATE = 16000
HOP_SIZE = 64
SECOND = SAMPLE_RATE // HOP_SIZE


def rescale(xs, min_val, max_val):
    norm = (xs - np.min(xs)) / (np.max(xs) - np.min(xs))
    return norm * (max_val - min_val) + min_val


def rescale_first_last(xs, min_val, max_val):
    norm = (xs - xs[0]) / (xs[-1] - xs[0])
    return norm * (max_val - min_val) + min_val


def fractal(points, n=3):
    points = np.array(points)
    segments = points.copy()
    for _ in range(n):
        segments = np.concatenate([rescale_first_last(points, f, s)[:-1] for f, s in zip(segments, segments[1:])])
        segments = np.concatenate((segments, points[-1:]))
    return segments


def to_delta(d):
    return d[1:] - d[:-1]


def from_delta(d):
    return np.concatenate([[0], np.cumsum(d)])


def fractal_bpc_floor(values, delta_times, n=0):
    values = fractal(values, n)
    delta_times = to_delta(fractal(from_delta(delta_times), n))

    return bpc_floor(values, delta_times), delta_times


def fractal_bpc(values, delta_times, n=0):
    values = fractal(values, n)
    delta_times = to_delta(fractal(from_delta(delta_times), n))

    return bpc(values, delta_times), delta_times


def time_to_step(duration):
    return int(duration * SECOND)


def get_t(duration):
    return np.linspace(0., duration, time_to_step(duration))


def phasor(duration, freq, phase=0.):
    t = freq * get_t(duration) + phase
    signal = t % 1.

    return signal


def sinusoid(duration, freq, center=0., amp=1., phase=0., power=1.):
    t = get_t(duration)
    w = freq * 2 * np.pi
    phase *= 2 * np.pi
    signal = center + amp * (np.sin(w * t + phase) ** power)

    return signal


def sweeping_sinusoid(duration, f0, f1, center=0., amp=1., phase=0., power=1.):
    t = get_t(duration)
    w0 = f0 * 2 * np.pi
    w1 = f1 * 2 * np.pi
    a = (w1 - w0) / duration
    b = w0
    phase *= 2 * np.pi
    signal = center + amp * (np.sin((a / 2) * (t ** 2) + b * t + phase) ** power)

    return signal


def constant(duration, value=0.):
    return np.ones(time_to_step(duration)) * value


def line_segment(duration, start=0., end=1.):
    t = get_t(duration)
    a = (end - start) / duration
    b = start

    return a * t + b


def sweeping_phasor(duration, f0, f1, phase=0.):
    t = get_t(duration)
    a = (f1 - f0) / duration
    b = f0

    t = (a / 2) * (t ** 2) + b * t + phase

    signal = t % 1.

    return signal


def triangle(duration, freq, ratio=0.5, lowest=0., highest=1., phase=0., power=1.):
    t = phasor(duration, freq, phase)

    signal = t / ratio
    mask = t > ratio
    signal[mask] = (t[mask] - 1) / (ratio - 1)

    signal **= power
    signal = rescale(signal, lowest, highest)

    return signal


def sweeping_triangle(duration, f0, f1, ratio=0.5, lowest=0., highest=1., phase=0., power=1.):
    t = sweeping_phasor(duration, f0, f1, phase)

    signal = t / ratio
    mask = t > ratio
    signal[mask] = (t[mask] - 1) / (ratio - 1)

    signal **= power
    signal = rescale(signal, lowest, highest)

    return signal


def square(duration, freq, ratio=0.5, lowest=0., highest=1., phase=0.):
    t = phasor(duration, freq, phase)
    signal = np.ones_like(t)
    signal[t > ratio] = 0.

    signal = rescale(signal, lowest, highest)

    return signal


def sweeping_square(duration, f0, f1, ratio=0.5, lowest=0., highest=1., phase=0.):
    t = sweeping_phasor(duration, f0, f1, phase)

    signal = np.ones_like(t)
    signal[t > ratio] = 0.

    signal = rescale(signal, lowest, highest)

    return signal


def adsr(a, d, s, r, peak, vol):
    attack = line_segment(a, 0., peak)
    decay = line_segment(d, peak, vol)
    sustain = constant(s, vol)
    release = line_segment(r, vol, 0)

    envelop = np.concatenate([
        attack, decay, sustain, release
    ])

    return envelop


def bpc(values, durations):
    result = []

    for idx, d in enumerate(durations):
        result.append(line_segment(d, values[idx], values[idx + 1]))

    return np.concatenate(result)


def bpc_floor(values, durations):
    result = []

    for idx, d in enumerate(durations):
        result.append(line_segment(d, values[idx], values[idx]))

    return np.concatenate(result)


def figure(pitches, durations):
    return np.concatenate([constant(d, p) for d, p in zip(durations, pitches)])


scales = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
    'harmonic_major': [0, 2, 4, 5, 7, 8, 11],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'diminished': [0, 2, 3, 5, 6, 8, 9, 11],
    'whole-tone': [0, 2, 4, 6, 8, 10],
    'augmented': [0, 3, 4, 7, 8, 11],
    'chromatic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}


def autotune(f0_midi, amount=0.0, scale='chromatic', rotate=0, offset=0):
    """Reduce variance of f0 from the chromatic or scale intervals."""
    if isinstance(scale, list):
        scale = deque(scale)
    else:
        scale = deque(scales[scale])
    scale.rotate(rotate)
    scale = np.array([(s - scale[0]) % 12 for s in scale])
    scale = np.concatenate([scale + 12 * i for i in range(10)])
    scale = (scale + offset) % 120

    return autotune_explicit(f0_midi, amount, scale)


def autotune_explicit(f0_midi, amount, scale):
    idx = np.abs(f0_midi[np.newaxis, :] - scale[:, np.newaxis]).argmin(axis=0)
    midi_diff = f0_midi - scale[idx]

    # Adjust the midi signal.
    return f0_midi - amount * midi_diff


def to_db_loudness(loudness, instrument, db_offset=0., min_loudness=0.0):
    model_dir = f'pretrained/{instrument.lower()}'
    with open(os.path.join(model_dir, 'dataset_statistics.pkl'), 'rb') as f:
        stats = pickle.load(f)
    qt = stats['quantile_transform']

    _, loudness_norm = fit_quantile_transform(
        loudness,
        loudness > min_loudness,
        inv_quantile=qt)

    loudness_norm = loudness_norm[:, 0]
    # silence silent parts
    loudness_norm[loudness <= min_loudness] -= 20
    return loudness_norm + db_offset


def generate_audio(instrument, pitch, loudness_db):
    af = {
        'f0_hz': librosa.midi_to_hz(pitch),
        'loudness_db': loudness_db
    }
    # Pretrained models.
    model_dir = f'pretrained/{instrument.lower()}'

    gin_file = os.path.join(model_dir, 'operative_config-0.gin')

    # Parse gin config,
    with gin.unlock_config():
        gin.parse_config_file(gin_file, skip_unknown=True)

    # Assumes only one checkpoint in the folder, 'ckpt-[iter]`.
    ckpt_files = [f for f in os.listdir(model_dir) if 'ckpt' in f]
    ckpt_name = ckpt_files[0].split('.')[0]
    ckpt = os.path.join(model_dir, ckpt_name)

    # Ensure dimensions and sampling rates are equal
    time_steps_train = gin.query_parameter('F0LoudnessPreprocessor.time_steps')
    n_samples_train = gin.query_parameter('Harmonic.n_samples')
    hop_size = int(n_samples_train / time_steps_train)

    time_steps = len(af['f0_hz'])
    n_samples = time_steps * hop_size

    gin_params = [
        'Harmonic.n_samples = {}'.format(n_samples),
        'FilteredNoise.n_samples = {}'.format(n_samples),
        'F0LoudnessPreprocessor.time_steps = {}'.format(time_steps),
        'oscillator_bank.use_angular_cumsum = True',  # Avoids cumsum accumulation errors.
    ]

    with gin.unlock_config():
        gin.parse_config(gin_params)

    # Set up the model just to predict audio given new conditioning
    instrument = ddsp.training.models.Autoencoder()
    instrument.restore(ckpt)

    # Build model by running a batch through it.
    outputs = instrument(af, training=False)
    audio_gen = instrument.get_audio_from_outputs(outputs).numpy()[0]
    del instrument
    return audio_gen


def scale_from_intervals(intervals):
    steps = cycle(intervals)
    scale = [0]

    while scale[-1] < 120:
        scale.append(scale[-1] + next(steps))

    return np.array(scale, dtype='int')


def harmonics_to_cents(harmonics):
    ratios = [b / a for a, b in zip(harmonics[:-1], harmonics[1:])]
    cents = [np.log2(x) * 12 for x in ratios]

    return cents


def random_adsr(duration):
    break_points = sorted(np.random.uniform(0, duration, 3))
    durs = np.diff(np.array([0, *break_points, duration]))
    env = adsr(*durs, 1, 0.5)
    steps = time_to_step(duration)
    if len(env) < steps:
        env = np.pad(env, (0, steps - len(env)))
    return env


def harmonics_to_pitches(base_pitch, harmonics):
    pitches = harmonics_to_cents(harmonics)
    pitches = accumulate(pitches, initial=0)
    pitches = [base_pitch + p for p in pitches]

    return pitches
