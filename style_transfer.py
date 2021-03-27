import librosa
import numpy as np
import soundfile
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F  # noqa


CONTENT_PATH = './contents/haiku.wav'
STYLE_PATH = './styles/389 Choralgesänge \'Chorale Harmonizations\', Part 1 - Bach (Score)-Khn9jLIYE4A.wav'
OUTPUT_PATH = './results/haiku-chorale.wav'


def normalize_audio(x):
    # Remove DC Offset
    x = x - x.mean()
    # Normalize
    x = x / np.max(np.abs(x))

    return x


def prepare_spectra(file_path, sample_rate, win_length, hop_length):
    audio, _ = librosa.load(file_path, sr=sample_rate, mono=True)
    audio, _ = librosa.effects.trim(audio)
    audio = normalize_audio(audio)
    complex_spectrum = librosa.stft(audio, n_fft=win_length, hop_length=hop_length)
    mag = np.abs(complex_spectrum)
    db = np.log1p(mag)

    # (channels, frames)
    return db, len(audio)


class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x


def gram_matrix(x):
    batch, channels, frames = x.size()
    features = x.view(batch * channels, frames)
    gram = torch.mm(features, features.t())

    return gram.div(batch * channels * frames)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, x):
        gram = gram_matrix(x)
        self.loss = F.mse_loss(gram, self.target)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, in_ch, out_ch, size):
        super().__init__()
        self.padding = (size - 1) // 2
        std = np.sqrt(2) * np.sqrt(2 / ((in_ch + out_ch) * size))
        kernel = torch.randn(out_ch, in_ch, size) * std
        self.register_buffer('conv_kernel', kernel)

    def forward(self, x):
        y = F.pad(x, (self.padding, self.padding))
        y = F.conv1d(y, self.conv_kernel)
        y = F.relu(y)

        return y


def main():
    print('Preparing files...')
    sr = 44100
    win_length = 2048
    hop_length = 256
    content, content_length = prepare_spectra(CONTENT_PATH, sr, win_length, hop_length)
    style, _ = prepare_spectra(STYLE_PATH, sr, win_length, hop_length)

    # Normalize spectra
    elem_mean = np.mean(content)
    elem_std = np.std(content)

    content = (content - elem_mean) / elem_std
    style = (style - elem_mean) / elem_std

    # trim to the shortest size
    length = min(content.shape[1], style.shape[1])
    offset = style.shape[1] // 8
    content, style = content[:, :length], style[:, offset:offset + length]

    n_channels = content.shape[0]

    content = torch.from_numpy(np.ascontiguousarray(content)).unsqueeze(0).cuda()
    style = torch.from_numpy(np.ascontiguousarray(style)).unsqueeze(0).cuda()

    print('Preparing network...')
    net = nn.Sequential(FeatureExtractor(n_channels, 4096, 17).cuda())

    with torch.no_grad():
        content_features = net(content)
        style_features = net(style)

    content_loss = ContentLoss(content_features)
    net.add_module('content_loss', content_loss)
    style_loss = StyleLoss(style_features)
    net.add_module('style_loss', style_loss)

    alpha = 1
    beta = 1e13
    lr = 1
    max_iter = 1000

    optimizer = torch.optim.LBFGS([content.requires_grad_()], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        net(content)
        loss = beta * net.style_loss.loss + alpha * net.content_loss.loss
        loss.backward()

        return loss

    print('Training...')
    optimizer.step(closure)

    net.cpu()
    style.cpu()
    del net
    del style

    print('Griffin-Lim...')
    with torch.no_grad():
        content = content * elem_std + elem_mean

        result = torch.exp(content) - 1
        result = torchaudio.functional.griffinlim(result,
                                                  window=torch.hann_window(win_length, True).cuda(),
                                                  n_fft=win_length,
                                                  hop_length=hop_length,
                                                  win_length=win_length,
                                                  power=1, n_iter=5000, momentum=0.99,
                                                  length=content_length,
                                                  rand_init=True)

    print('Writing result to file...')
    result = result.cpu().numpy()[0]
    result = normalize_audio(result)
    soundfile.write(OUTPUT_PATH, result, sr)


if __name__ == '__main__':
    main()
