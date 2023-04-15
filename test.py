import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from train import prepare_dataloaders, load_model, load_checkpoint
from hparams import create_hparams
import argparse
import numpy as np
from loss_function import Tacotron2Loss, Iso_Tacotron2Loss
import soundfile as sf
import librosa
from matplotlib import pyplot as plt
import os


def test(checkpoint_path, isochronic=False):
    """Run Inference"""
    _, _, testset, collate_fn = prepare_dataloaders(hparams)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate,
                                 weight_decay=hparams.weight_decay)
    print(isochronic)
    if isochronic:
        criterion = Iso_Tacotron2Loss()
    else:
        criterion = Tacotron2Loss()

    model, _, _, _ = load_checkpoint(checkpoint_path, model, optimizer)

    model.eval()
    with torch.no_grad():
        test_loader = DataLoader(testset, sampler=None, num_workers=1,
                                 shuffle=False, batch_size=1,
                                 pin_memory=False, collate_fn=collate_fn, drop_last=False)

        test_loss = 0.0
        for i, batch in enumerate(test_loader):
            x, y = model.parse_batch(batch)
            # x is inputs, input_lengths, mels, max_len, output_lengths
            y_pred = model(x)
            print("Y_PRED[1] shape", y_pred[1].shape)
            out_sig = y_pred[1].detach().cpu().numpy()

            out_sig = np.squeeze(out_sig)
            spectrogram = np.abs(out_sig)
            # power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
            # Show Spectrogram
            plt.figure(figsize=(16, 5))
            librosa.display.specshow(spectrogram, sr=hparams.sampling_rate, x_axis='time', y_axis='mel', cmap='magma',
                                     hop_length=hparams.hop_length)
            plt.colorbar(label='dB')
            plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
            plt.xlabel('Time', fontdict=dict(size=15))
            plt.ylabel('Frequency', fontdict=dict(size=15))
            plt.savefig(f"Output/spectogram{i}.png") # Has correct frequency Range

            # Convert Amplitude to Decibels
            log_spectro = librosa.amplitude_to_db(spectrogram)
            plt.figure(figsize=(16, 5))
            librosa.display.specshow(log_spectro, sr=hparams.sampling_rate, x_axis='time', y_axis='hz',
                                     hop_length=hparams.hop_length, cmap='magma')
            plt.colorbar(label='Decibels')
            plt.title('Spectrogram (dB)', fontdict=dict(size=18))
            plt.xlabel('Time', fontdict=dict(size=15))
            plt.ylabel('Frequency', fontdict=dict(size=15))
            plt.savefig(f"Output/spect_amp2db{i}.png") # Incorrect frequency Range

            # Convert Power to db
            power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
            plt.figure(figsize=(16, 5))
            librosa.display.specshow(power_to_db, sr=hparams.sampling_rate, x_axis='time', y_axis='mel', cmap='magma',
                                     hop_length=hparams.hop_length)
            plt.colorbar(label='dB')
            plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
            plt.xlabel('Time', fontdict=dict(size=15))
            plt.ylabel('Frequency', fontdict=dict(size=15))
            plt.savefig(f"Output/spect_power2db{i}.png") # Correct frequency range


            # print(out_sig.shape)
            # for j in range(4):
            #   print(f"Saving figure: {j}")
            #   plt.figure()
            #   # librosa.display.specshow(librosa.power_to_db(S, ref=np.max),ax=ax)
            #   plt.imshow(out_sig[j])
            #   plt.colorbar()
            #   # plt.show()
            #   plt.savefig(os.path.join(output_directory,f'plot_{iteration}_{j}.png'))
            #   # out_ = librosa.feature.inverse.mel_to_audio(out_sig[j])
            #   # sf.write(os.path.join(output_directory,f"out_{iteration}.wav"),out_,22050)

            loss = criterion(y_pred, y)
            reduced_test_loss = loss.item()
            test_loss += reduced_test_loss

            # Save output audio
            # np.save(f"Output/{i}", out_sig)
            # print(np.shape(np.squeeze(out_sig)))
            # # print("out_sig", out_sig)
            # sf.write(f"Output/{i}.wav", np.squeeze(out_sig), 22050, subtype='FLOAT') # TODO: Check Sampling Rate
            # # sf.write(f"Output/{i}.flac", np.squeeze(out_sig), 22050, format='FLAC')
            # y = librosa.feature.inverse.mel_to_audio(np.squeeze(out_sig))
            # print("y", y)
            # sf.write(f'Output/test{i}.wav', y, 22050)
            # sf.write(f'Output/test{i}_float.wav', y, 22050, subtype='FLOAT')
            # sf.write(f'Output/test{i}.flac', y, 22050, format='FLAC')

            if i == 3:
                break
        test_loss = test_loss / (i + 1)

    return test_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('-i', '--isochronic', action='store_true',
                        help='For training isochrony')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    loss = test(args.checkpoint_path, isochronic=args.isochronic)
    print(loss)
