import numpy as np
import torch
import torch.utils.data
import torchaudio.transforms as T
import librosa
import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from audio_processing import dynamic_range_compression


class MelLoader(torch.utils.data.Dataset):
    """
        1) loads audio pairs
        3) computes mel-spectrograms from audio files.
    """

    def __init__(self, audiopaths, hparams):
        self.inputs = audiopaths[0]
        self.outputs = audiopaths[1]
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        self.resampler = T.Resample(44100, hparams.sampling_rate)
        self.melspectrogram = T.MelSpectrogram(sample_rate=hparams.sampling_rate,
                                               n_fft=hparams.filter_length,
                                               win_length=hparams.win_length,
                                               hop_length=hparams.hop_length,
                                               n_mels=80,
                                               center=True,
                                               pad_mode="reflect",
                                               power=2.0)

    def get_mel_pair(self, index):
        inputs = self.get_mel(self.inputs[index])
        outputs = self.get_mel(self.outputs[index])

        # Normalize
        inputs = dynamic_range_compression(inputs, .2)
        outputs = dynamic_range_compression(outputs, .2)

        return inputs, outputs

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.sampling_rate:
                # Down sample
                if sampling_rate == 44100:
                    audio = self.resampler(audio)
                else:
                    resampler2 = T.Resample(sampling_rate, self.sampling_rate)
                    audio = resampler2(audio)

            mel_spec = self.melspectrogram(audio)
            mel_spec = torch.squeeze(mel_spec, 0)

        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return mel_spec

    def __getitem__(self, index):
        return self.get_mel_pair(index)

    def __len__(self):
        return len(self.inputs)


class MelCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """

    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collates training batch from mel-spectrograms
        PARAMS
        ------
        batch: [input mel_normalized, output mel_normalized]
        """

        # Right zero-pad mel-spec
        num_mels = batch[0][0].size(0)
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]
        if max_input_len % self.n_frames_per_step != 0:
            max_input_len += self.n_frames_per_step - max_input_len % self.n_frames_per_step
            assert max_input_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        input_padded = torch.FloatTensor(len(batch), num_mels, max_input_len)
        input_padded.zero_()
        # gate_padded = torch.FloatTensor(len(batch), max_target_len)
        # gate_padded.zero_()
        for i in ids_sorted_decreasing:
            mel = batch[i][0]
            input_padded[i, :, :mel.size(1)] = mel
            # gate_padded[i, mel.size(1)-1:] = 1

        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in ids_sorted_decreasing:
            mel = batch[i][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1:] = 1
            output_lengths[i] = mel.size(1)

        return input_padded, input_lengths, mel_padded, gate_padded, output_lengths
