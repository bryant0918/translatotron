from matplotlib import pyplot as plt
import numpy as np
import librosa
import librosa.display
import torch
import torchaudio
import torchaudio.transforms as T
from audio_processing import dynamic_range_compression
from audio_processing import dynamic_range_decompression

output_file = "C:\\Users\\bryan\\Documents\\School\\Winter 2023\\CS 601R\\Final Project\\Data\\LibriS2S\\EN\\9\\sentence_level_audio\\00001-f000001.wav"

waveform, sr = torchaudio.load(output_file)
print(sr)
print(np.max(waveform.numpy()))
print(np.min(waveform.numpy()))

# Downsampled
resampler = T.Resample(44100, 22050, dtype=waveform.dtype)
resampled_waveform = resampler(waveform)
sr = 22050

# torchaudio.save("Output/resampled_de.wav", resampled_waveform, sr)
waveform = resampled_waveform

# output_file = "C:\\Users\\bryan\\Documents\\School\\Winter 2023\\CS 601R\\Final Project\\Data\\LibriS2S\\DE\\9\\sentence_level_audio\\00001-das_bildnis_4.flac"
#
# waveform, sr = torchaudio.load(output_file)
# print(sr)
# print(np.max(waveform.numpy()))
# print(np.min(waveform.numpy()))


def plot_waveform(waveform, sr, title="Waveform"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    figure, axes = plt.subplots(num_channels, 1)
    axes.plot(time_axis, waveform[0], linewidth=1)
    axes.grid(True)
    figure.suptitle(title)
    plt.show()


plot_waveform(waveform, sr)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    """Plot various spectrograms"""
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.savefig(title) if title else plt.show()


def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    """Plot spectrogram from waveform"""
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    print(num_channels, num_frames)
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show()


# plot_specgram(waveform, sample_rate=sr)

n_fft = 1024
win_length = None
hop_length = 512

# Define transform
melspectrogram = T.MelSpectrogram(sample_rate=sr,
                                  n_fft=n_fft,
                                  win_length=win_length,
                                  hop_length=hop_length,
                                  n_mels=80,
                                  center=True,
                                  pad_mode="reflect")

mel_spec = melspectrogram(waveform)
# print(mel_spec.shape)
# print(mel_spec[0].shape)
plot_spectrogram(mel_spec[0])

# print(np.max(mel_spec.numpy()))
# print(np.min(mel_spec.numpy()))

mel_spec = dynamic_range_compression(mel_spec, .2)
# plot_spectrogram(mel_spec[0], title=f"Output/spectrogram_compressed.png")
# plot_spectrogram(mel_spec[0])

# print("After dynamic range compression")
# print(np.max(mel_spec.numpy()))
# print(np.min(mel_spec.numpy()))

mel_spec = dynamic_range_decompression(mel_spec, .2)
# plot_spectrogram(mel_spec[0], title=f"Output/spectrogram_decompressed.png")
plot_spectrogram(mel_spec[0])

# print("After dynamic range decompression")
# print(np.max(mel_spec.numpy()))
# print(np.min(mel_spec.numpy()))

# linear_to_decibel(1 + abs(melspectrogram))

power_spec = T.InverseMelScale(n_stft=1024 // 2 + 1,
                               n_mels=80,
                               sample_rate=sr,
                               f_min=0.0)

powr_spec = power_spec(mel_spec)
print("Got powr_spec")

plot_spectrogram(powr_spec[0])
# print(np.max(mel_spec.numpy()))
# print(np.min(mel_spec.numpy()))


griffin_lim = T.GriffinLim(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    power=2)

reconstructed_waveform = griffin_lim(powr_spec)
print("Got reconstruction")
print(reconstructed_waveform.shape)

plot_waveform(reconstructed_waveform, sr, title="reconstructed")

torchaudio.save("Demo/00001-f000001_synth.wav", reconstructed_waveform, sr)
