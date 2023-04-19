import torch
from train import load_model, load_checkpoint
from hparams import create_hparams
from loss_function import Tacotron2Loss, Iso_Tacotron2Loss
from plotting_utils import save_spectrogram
import torchaudio
import torchaudio.transforms as T
from audio_processing import dynamic_range_compression, dynamic_range_decompression
import os

hparams = create_hparams()


def infer(checkpoint_path, audiofile, out_dir):
    # Define needed transformations
    resampler = T.Resample(44100, hparams.sampling_rate)

    melspectrogram = T.MelSpectrogram(sample_rate=hparams.sampling_rate,
                                      n_fft=hparams.filter_length,
                                      win_length=hparams.win_length,
                                      hop_length=hparams.hop_length,
                                      n_mels=hparams.n_mel_channels,
                                      center=True,
                                      pad_mode="reflect",
                                      power=2.0)

    power_spec = T.InverseMelScale(n_stft=hparams.filter_length // 2 + 1,
                                   n_mels=hparams.n_mel_channels,
                                   sample_rate=hparams.sampling_rate,
                                   f_min=0.0)

    griffin_lim = T.GriffinLim(n_fft=hparams.filter_length,
                               win_length=hparams.win_length,
                               hop_length=hparams.hop_length,
                               power=2)

    # Preprocess
    audio, sampling_rate = torchaudio.load(audiofile)
    assert sampling_rate == 22050
    mel_spec = melspectrogram(audio)
    mel_spec_normed = dynamic_range_compression(mel_spec)

    model = load_model(hparams)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate,
                                 weight_decay=hparams.weight_decay)

    model, _, _, _ = load_checkpoint(checkpoint_path, model, optimizer)

    """Run Inference"""
    model.eval()
    with torch.no_grad():
        mel_pred, mel_postnet, gate, alignments = model.inference(mel_spec_normed[0])
        print("mel_pred shape", mel_pred.shape)

        # Get denormalized signal for visualization and audio
        out_sig = dynamic_range_decompression(mel_pred, .2)

        # Show or save Spectrogram
        save_spectrogram(out_sig, os.path.join(out_dir, "mel_spec.png"))

        # Get waveform from spectrogram
        powr_spec = power_spec(out_sig)
        reconstructed_waveform = griffin_lim(powr_spec)

        # Save output audio
        torchaudio.save(os.path.join(out_dir, "predicted_audio.wav"), reconstructed_waveform, hparams.sampling_rate)

    return


if __name__ == "__main__":
    ckpt_path = "C:\\Users\\bryan\\Documents\\School\\Winter 2023\\CS 601R\\Final Project\\ckpts\\checkpoint_970"
    audiofile = "C:\\Users\\bryan\\Documents\\School\\Winter 2023\\CS 601R\\Final Project\\Data\\LibriS2S\\DE\\9\\sentence_level_audio\\00001-das_bildnis_4.flac"
    out_dir = "C:\\Users\\bryan\\Documents\\School\\Winter 2023\\CS 601R\\Final Project\\Output\\inference"

    infer(ckpt_path, audiofile, out_dir)
