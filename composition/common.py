import numpy as np

sample_rate = 16000
hop_size = 64
step_dur = hop_size / sample_rate


def oscillate(t, freq, center, amp, phase=0., power=1):
    return center + amp * np.sin(freq * 2 * np.pi * (t + phase)) ** power


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
    factor = dur / sum([a, d, s, r])
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
