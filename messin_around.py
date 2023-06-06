import soundfile as sf
from matplotlib import pyplot as plt
import numpy as np
import librosa
import librosa.display
import torch

print(sf.available_subtypes('wav'))
print(sf.available_subtypes('flac'))

output_file = "C:\\Users\\bryan\\Documents\\School\\Winter 2023\\CS 601R\\Final Project\\Data\\LibriS2S\\EN\\9\\sentence_level_audio\\00001-f000009.wav"
input_file = "C:\\Users\\bryan\\Documents\\School\\Winter 2023\\CS 601R\\Final Project\\Data\\LibriS2S\\DE\\9\\sentence_level_audio\\00001-das_bildnis_4.flac"

data1, sr1 = sf.read(output_file)
print(sr1)

# Downsampled
data1 = librosa.resample(data1, orig_sr=sr1, target_sr=22050)
sr1 = 22050

# # Show waveplot
# plt.figure(figsize=(16, 5))
# librosa.display.waveshow(data1, sr=sr1)
# plt.title("Waveplot", fontdict=dict(size=18))
# plt.xlabel("Time", fontdict=dict(size=15))
# plt.ylabel("Amplitude", fontdict=dict(size=15))
# # plt.show()

# Get Spectrogram (amplitude)
hop_length=512
win_length=1024
n_fft = 1024
audio_stft = librosa.core.stft(data1, hop_length=hop_length, n_fft=n_fft)
spectrogram = np.abs(audio_stft)

# # Show Amplitude Spectogram
# plt.figure(figsize=(16, 5))
# librosa.display.specshow(spectrogram, sr=sr1, x_axis='time', y_axis='hz', hop_length=hop_length)
# plt.colorbar(label="Amplitude")
# plt.title("Spectrogram (amp)", fontdict=dict(size=18))
# plt.xlabel("Time", fontdict=dict(size=15))
# plt.ylabel("Frequency", fontdict=dict(size=15))
# plt.show()

# Convert Amplitude to Decibels
log_spectro = librosa.amplitude_to_db(spectrogram)

# # Show Spectrogram in (DB)
# plt.figure(figsize=(16, 5))
# librosa.display.specshow(log_spectro, sr=sr1, x_axis='time', y_axis='hz', hop_length=hop_length, cmap='magma')
# plt.colorbar(label='Decibels')
# plt.title('Spectrogram (dB)', fontdict=dict(size=18))
# plt.xlabel('Time', fontdict=dict(size=15))
# plt.ylabel('Frequency', fontdict=dict(size=15))
# # plt.show()

# Get Mel Signal
mel_signal = librosa.feature.melspectrogram(y=data1, sr=sr1, hop_length=hop_length,
 n_fft=n_fft, n_mels=80)
spectrogram = np.abs(mel_signal)
power_to_db = librosa.power_to_db(spectrogram, ref=np.max)

# Show Mel-Spectrogram
plt.figure(figsize=(16, 5))
librosa.display.specshow(power_to_db, sr=sr1, x_axis='time', y_axis='mel', cmap='magma',
 hop_length=hop_length)
plt.colorbar(label='dB')
plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
plt.xlabel('Time', fontdict=dict(size=15))
plt.ylabel('Frequency', fontdict=dict(size=15))
plt.savefig(f"Output/spectrogram_actual.png")

# Convert Mel-Spectrogram back to Audio
print("SHAPE", np.shape(mel_signal))
print(np.shape(torch.tensor(mel_signal).unsqueeze(0)))
y = librosa.feature.inverse.mel_to_audio(mel_signal)
print("y", np.shape(y))
print(y[20000:20100])
sf.write('Output/00001-f000009_synth.wav', y, sr1)

# # Convert Mel-Spectrogram back to Audio with WaveGLow
# print("Wrote output wav and loading Waveglow")
# waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
# print("Loaded Waveglow")
# waveglow = waveglow.remove_weightnorm(waveglow)
# waveglow = waveglow.to('cuda')
# waveglow.eval()
#
# with torch.no_grad():
#     audio = waveglow.infer(torch.tensor(mel_signal).unsqueeze(0).to('cuda', dtype=torch.float))
# audio_numpy = audio[0].data.cpu().numpy()
# sf.write('Output/waveglow.wav', audio_numpy, sr1)

# English Audio
data2, sr2 = sf.read(input_file)
print(sr2)

out_sig = np.load("Output/2.npy")
print(out_sig.shape)

# with torch.no_grad():
#     audio = waveglow.infer(torch.tensor(out_sig).to('cuda', dtype=torch.float))
# audio_numpy = audio[0].data.cpu().numpy()
# sf.write('Output/waveglow2.wav', audio_numpy, sr1)

# for i in range(5):
#     print(i)
#     out_sig = np.load(f"Output/{i}.npy")
#     out_sig = np.squeeze(out_sig)
#
#     # Show Mel-Spectrogram
#     spectrogram = np.abs(out_sig)
#     power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
#     plt.figure(figsize=(16, 5))
#     librosa.display.specshow(power_to_db, sr=sr1, x_axis='time', y_axis='mel', cmap='magma',
#                              hop_length=hop_length)
#     plt.colorbar(label='dB')
#     plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
#     plt.xlabel('Time', fontdict=dict(size=15))
#     plt.ylabel('Frequency', fontdict=dict(size=15))
#     plt.show()
#
#     y = librosa.feature.inverse.mel_to_audio(out_sig)
#     sf.write(f'Output/mel_to_audio{i}.wav', y, sr1)
#
# S = librosa.feature.melspectrogram(
#     y=data2,
#     sr=sr2,
#     win_length=1024,
#     hop_length=256,
#     n_mels=80,
#     fmin= 0.0,
#     fmax=8000.0)


# print(f"Showing figure")
# plt.figure()
# fig, ax = plt.subplots()
# img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max), ax=ax)
# fig.colorbar(img, ax=ax, format='%+2.0f dB')
# ax.set(title='Mel-frequency spectrogram')

# plt.imshow(out_sig)
# plt.colorbar()
# plt.show()
# plt.savefig(os.path.join(output_directory, f'plot_{iteration}_{j}.png'))
# out_ = librosa.feature.inverse.mel_to_audio(S)
# sf.write(f"Output/out.wav", out_, 44100)


# Torch's Mel Spectrogram Stuff



