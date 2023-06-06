import soundfile as sf
from matplotlib import pyplot as plt
import numpy as np
import librosa
import librosa.display
import torch
import layers
from utils import load_wav_to_torch
from scipy.io import wavfile

output_file = "C:\\Users\\bryan\\Documents\\School\\Winter 2023\\CS 601R\\Final Project\\Data\\LibriS2S\\EN\\9\\sentence_level_audio\\00001-f000009.wav"
input_file = "C:\\Users\\bryan\\Documents\\School\\Winter 2023\\CS 601R\\Final Project\\Data\\LibriS2S\\DE\\9\\sentence_level_audio\\00001-das_bildnis_4.flac"

data1, sr1 = sf.read(output_file)

# Downsampled
data1 = librosa.resample(data1, orig_sr=sr1, target_sr=22050)
sr1 = 22050

# # Show waveplot
# plt.figure(figsize=(16, 5))
# librosa.display.waveshow(data1, sr=sr1)
# plt.title("Waveplot", fontdict=dict(size=18))
# plt.xlabel("Time", fontdict=dict(size=15))
# plt.ylabel("Amplitude", fontdict=dict(size=15))
# plt.show()

# Set up for Mel
hop_length=512
win_length=1024
n_fft = 1024
filter_length = 1024
n_mel_channels = 80
sampling_rate = 44100
mel_fmin = 0.0
mel_fmax = 8000.0

stft = layers.TacotronSTFT(filter_length, hop_length, win_length,
                n_mel_channels, sampling_rate, mel_fmin, mel_fmax)


def get_mel(filename):
    audio, sampling_rate = load_wav_to_torch(filename)
    if sampling_rate != stft.sampling_rate:
        # self.stft.sampling_rate = sampling_rate
        # Downsample
        audio = librosa.resample(audio.numpy(), orig_sr=sampling_rate, target_sr=stft.sampling_rate)
        audio = torch.tensor(audio)

    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)

    return melspec


# Get Mel Signal
mel_signal = get_mel(output_file)
mel_signal = mel_signal.numpy()
spectrogram = np.abs(mel_signal)
power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
# Convert Amplitude to Decibels
log_spectro = librosa.amplitude_to_db(spectrogram)

print(np.min(mel_signal))
print(np.max(mel_signal))

sf.write("Output/raw.wav", mel_signal, 44100)

# # Show Mel-Spectrogram mel_signal
# plt.figure(figsize=(16, 5))
# librosa.display.specshow(mel_signal, sr=sr1, x_axis='time', y_axis='mel', cmap='magma',
#  hop_length=hop_length)
# plt.colorbar(label='dB')
# plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
# plt.xlabel('Time', fontdict=dict(size=15))
# plt.ylabel('Frequency', fontdict=dict(size=15))
# plt.show() # Looks the best
#
# Show Mel-Spectrogram mel_signal
a1, a2 = np.min(mel_signal), np.max(mel_signal)
b1, b2 = -80, 0
mel_sig = b1 + (mel_signal - a1)*(b2-b1)/(a2-a1)

# plt.figure(figsize=(16, 5))
# librosa.display.specshow(mel_sig, sr=sr1, x_axis='time', y_axis='mel', cmap='magma',
#  hop_length=hop_length)
# plt.colorbar(label='dB')
# plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
# plt.xlabel('Time', fontdict=dict(size=15))
# plt.ylabel('Frequency', fontdict=dict(size=15))
# plt.show() # Looks the best

y = librosa.feature.inverse.mel_to_audio(mel_signal, n_fft=1024)
print("y", np.shape(y))
print(y[20000:20100])

sf.write('Output/mel_signal.wav', y, sampling_rate)
# print(np.shape(mel_sig.T))

# with wave.open("Output/sound1.wav", "w") as f:
#     f.setnchannels(1)
#     # 2 bytes per sample.
#     f.setsampwidth(2)
#     f.setframerate(sr1)
#     f.writeframes(mel_sig.tobytes())

#
# # Show Mel-Spectrogram spectrogram
# plt.figure(figsize=(16, 5))
# librosa.display.specshow(spectrogram, sr=sr1, x_axis='time', y_axis='mel', cmap='magma',
#  hop_length=hop_length)
# plt.colorbar(label='dB')
# plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
# plt.xlabel('Time', fontdict=dict(size=15))
# plt.ylabel('Frequency', fontdict=dict(size=15))
# plt.show()
#
# # Show Mel-Spectrogram power_to_db
# plt.figure(figsize=(16, 5))
# librosa.display.specshow(power_to_db, sr=sr1, x_axis='time', y_axis='mel', cmap='magma',
#  hop_length=hop_length)
# plt.colorbar(label='dB')
# plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
# plt.xlabel('Time', fontdict=dict(size=15))
# plt.ylabel('Frequency', fontdict=dict(size=15))
# plt.show()

# # Show Mel-Spectrogram amp_to_db
# plt.figure(figsize=(16, 5))
# librosa.display.specshow(log_spectro, sr=sr1, x_axis='time', y_axis='mel', cmap='magma',
#  hop_length=hop_length)
# plt.colorbar(label='dB')
# plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
# plt.xlabel('Time', fontdict=dict(size=15))
# plt.ylabel('Frequency', fontdict=dict(size=15))
# plt.show()

# Convert Mel-Spectrogram back to Audio
def get_audio(melspec):
    """Convert mel spectrogram to audio"""
    print("Melspec", melspec)
    mag, phase = librosa.magphase(melspec)
    print("Mag", mag)
    print("Phase", phase)
    denorm_mel = stft.spectral_de_normalize(ma)
    magnitude = None
    phase = None
    inv_transform = stft.stft_fn.inverse(magnitude, phase)
    return


# get_audio(mel_signal)


# German Audio
data2, sr2 = sf.read(input_file)
print(sr2)

out_sig = np.load("Output/2.npy")
print(out_sig.shape)

# Torch's Mel Spectrogram Stuff
