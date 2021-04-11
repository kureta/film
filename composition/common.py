from collections import deque

import numpy as np

sample_rate = 16000
hop_size = 64
step_dur = hop_size / sample_rate


def oscillate(t, freq, center, amp, phase=0., power=1):
    w = freq * 2 * np.pi
    return center + amp * np.sin(w * t + phase) ** power


def linear_oscillate(t, f0, f1, center, amp, phase=0., power=1):
    w0 = f0 * 2 * np.pi
    w1 = f1 * 2 * np.pi
    a = (w1 - w0) / t.max()
    b = w1
    return center + amp * np.sin((a / 2) * (t ** 2) + b * t + phase) ** power


def perc(attack_ratio=0.1, duration=0.5, peak=1.0):
    num_steps = int(duration / step_dur)
    attack_steps = int(num_steps * attack_ratio)
    decay_steps = num_steps - attack_steps

    result = np.zeros(num_steps)
    attack = np.linspace(0., peak, attack_steps)
    decay = np.linspace(peak, 0., decay_steps)

    result[:attack_steps] = attack
    result[-decay_steps:] = decay

    return result


def adsr(a, d, s, r, dur, peak, vol):
    factor = sum([a, d, s, r])
    num_steps = int(dur / step_dur)

    attack_steps = int(num_steps * a / factor)
    decay_steps = int(num_steps * d / factor)
    sustain_steps = int(num_steps * s / factor)
    release_steps = num_steps - (attack_steps + decay_steps + sustain_steps)

    attack = np.linspace(0., peak, attack_steps)
    decay = np.linspace(peak, vol, decay_steps)
    sustain = np.linspace(vol, vol, sustain_steps)
    release = np.linspace(vol, 0., release_steps)

    return np.concatenate([attack, decay, sustain, release])


def silence(duration=0.5):
    return np.zeros(int(duration / step_dur))


def rescale(xs, min_val, max_val):
    return xs * (max_val - min_val) + min_val


scales = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
    'harmonic_major': [0, 2, 4, 5, 7, 8, 11],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'diminished': [0, 2, 3, 5, 6, 8, 9, 11],
    'wholetone': [0, 2, 4, 6, 8, 10],
    'augmented': [0, 3, 4, 7, 8, 11],
    'chromatic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}


def autotune(f0_midi, amount=0.0, scale='chromatic', rotate=0, offset=0):
    """Reduce variance of f0 from the chromatic or scale intervals."""
    scale = deque(scales[scale])
    scale.rotate(rotate)
    scale = np.array([(s - scale[0]) % 12 for s in scale])
    scale = np.concatenate([scale + 12 * i for i in range(10)])
    scale = (scale + offset) % 120

    idx = np.abs(f0_midi[np.newaxis, :] - scale[:, np.newaxis]).argmin(axis=0)
    midi_diff = f0_midi - scale[idx]

    # Adjust the midi signal.
    return f0_midi - amount * midi_diff
