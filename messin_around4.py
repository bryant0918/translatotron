import os
import torchaudio
import torchaudio.transforms as T
import numpy as np

n_fft = 1024
win_length = None
hop_length = 512

output_data_root = 'C:\\Users\\bryan\\Documents\\School\\Winter 2023\\CS 601R\\Final Project\\Data\\LibriS2S\\EN\\9\\sentence_level_audio\\'
input_data_root = 'C:\\Users\\bryan\\Documents\\School\\Winter 2023\\CS 601R\\Final Project\\Data\\LibriS2S\\DE\\9\\sentence_level_audio\\'


# Define transforms
resampler = T.Resample(44100, 22050)
melspectrogram = T.MelSpectrogram(sample_rate=22050,
                                  n_fft=n_fft,
                                  win_length=win_length,
                                  hop_length=hop_length,
                                  n_mels=80,
                                  center=True,
                                  pad_mode="reflect")

# English
max_val = 0
max_wav = 0
for f in os.listdir(output_data_root):
    file = os.path.join(output_data_root, f)
    waveform, sr = torchaudio.load(file)
    if sr != 44100:
        print("SR: ", sr)
    resampler = T.Resample(44100, 22050)
    waveform = resampler(waveform)
    mel_spec = melspectrogram(waveform)
    if np.max(waveform.numpy()) > max_wav:
        max_wav = np.max(waveform.numpy())
    if np.max(mel_spec.numpy()) > max_val:
        max_val = np.max(mel_spec.numpy())
        print(max_val)

print("Final", max_val)
print("Final wav_val", max_wav)

# German
max_val = 0
max_wav = 0
for f in os.listdir(input_data_root):
    file = os.path.join(input_data_root, f)
    waveform, sr = torchaudio.load(file)
    if sr != 22050:
        print("SR: ", sr)
    mel_spec = melspectrogram(waveform)
    if np.max(waveform.numpy()) > max_wav:
        max_wav = np.max(waveform.numpy())
    if np.max(mel_spec.numpy()) > max_val:
        max_val = np.max(mel_spec.numpy())
        print(max_val)

print("Final", max_val)
print("Final wav_val", max_wav)
