from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from ddsp import core, spectral_ops


def specplot(audio,
             vmin=-5,
             vmax=1,
             rotate=True,
             size=512 + 256,
             **matshow_kwargs):
    """Plot the log magnitude spectrogram of audio."""
    # If batched, take first element.
    if len(audio.shape) == 2:
        audio = audio[0]

    logmag = spectral_ops.compute_logmag(core.tf_float32(audio), size=size)
    if rotate:
        logmag = np.rot90(logmag)
    # Plotting.
    plt.figure(figsize=(12, 6))
    plt.matshow(logmag,
                vmin=vmin,
                vmax=vmax,
                cmap=plt.cm.magma,
                aspect='auto',
                fignum=1,
                **matshow_kwargs)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Time')
    plt.ylabel('Frequency')


def get_tuning_factor(f0_midi, f0_confidence, mask_on):
    """Get an offset in cents, to most consistent set of chromatic intervals."""
    # Difference from midi offset by different tuning_factors.
    tuning_factors = np.linspace(-0.5, 0.5, 101)  # 1 cent divisions.
    midi_diffs = (f0_midi[mask_on][:, np.newaxis] -
                  tuning_factors[np.newaxis, :]) % 1.0
    midi_diffs[midi_diffs > 0.5] -= 1.0
    weights = f0_confidence[mask_on][:, np.newaxis]

    # Computes mininmum adjustment distance.
    cost_diffs = np.abs(midi_diffs)
    cost_diffs = np.mean(weights * cost_diffs, axis=0)

    # Computes mininmum "note" transitions.
    f0_at = f0_midi[mask_on][:, np.newaxis] - midi_diffs
    f0_at_diffs = np.diff(f0_at, axis=0)
    deltas = (f0_at_diffs != 0.0).astype(np.float)
    cost_deltas = np.mean(weights[:-1] * deltas, axis=0)

    # Tuning factor is minimum cost.
    norm = lambda x: (x - np.mean(x)) / np.std(x)
    cost = norm(cost_deltas) + norm(cost_diffs)
    return tuning_factors[np.argmin(cost)]


scales = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
    'harmonic_major': [0, 2, 4, 5, 7, 8, 11],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'diminished': [0, 2, 3, 5, 6, 8, 9, 11],
    'wholetone': [0, 2, 4, 6, 8, 10],
    'augmented': [0, 3, 4, 7, 8, 11],
}


def auto_tune_(f0_midi, tuning_factor, mask_on, amount=0.0, scale='chromatic', rotate=0):
    """Reduce variance of f0 from the chromatic or scale intervals."""
    if scale == 'chromatic':
        midi_diff = (f0_midi - tuning_factor) % 1.0
        midi_diff[midi_diff > 0.5] -= 1.0
    else:
        scale = deque(scales[scale])
        print(list(scale))
        scale.rotate(rotate)
        scale = [(s - scale[0]) % 12 for s in scale]
        print(scale)
        major_scale = np.ravel(
            [np.array(scale) + 12 * i for i in range(10)])
        all_scales = np.stack([major_scale + i for i in range(12)])

        f0_on = f0_midi[mask_on]
        # [time, scale, note]
        f0_diff_tsn = (
                f0_on[:, np.newaxis, np.newaxis] - all_scales[np.newaxis, :, :])
        # [time, scale]
        f0_diff_ts = np.min(np.abs(f0_diff_tsn), axis=-1)
        # [scale]
        f0_diff_s = np.mean(f0_diff_ts, axis=0)
        scale_idx = np.argmin(f0_diff_s)
        scale = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb',
                 'G', 'Ab', 'A', 'Bb', 'B', 'C'][scale_idx]

        # [time]
        f0_diff_tn = f0_midi[:, np.newaxis] - all_scales[scale_idx][np.newaxis, :]
        note_idx = np.argmin(np.abs(f0_diff_tn), axis=-1)
        midi_diff = np.take_along_axis(
            f0_diff_tn, note_idx[:, np.newaxis], axis=-1)[:, 0]
        print('Autotuning... \nInferred key: {}  '
              '\nTuning offset: {} cents'.format(scale, int(tuning_factor * 100)))

    # Adjust the midi signal.
    return f0_midi - amount * midi_diff


def autotune(f0_midi, amount, scale='chromatic', rotate=0):
    tuning_factor = get_tuning_factor(f0_midi, np.ones_like(f0_midi), np.ones_like(f0_midi, dtype='bool'))
    tuned = auto_tune_(f0_midi, tuning_factor, np.ones_like(f0_midi, dtype='bool'),
                       amount=amount,
                       scale=scale,
                       rotate=rotate
                       )

    return tuned
