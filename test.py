import torch
from torch.utils.data import DataLoader
from train import prepare_dataloaders, load_model, load_checkpoint
from hparams import create_hparams
import argparse
from loss_function import Tacotron2Loss, Iso_Tacotron2Loss
from plotting_utils import save_spectrogram
import torchaudio
import torchaudio.transforms as T
from audio_processing import dynamic_range_decompression


def test(checkpoint_path, isochronic=False):
    power_spec = T.InverseMelScale(n_stft=hparams.filter_length // 2 + 1,
                                   n_mels=hparams.n_mel_channels,
                                   sample_rate=hparams.sampling_rate,
                                   f_min=0.0)

    griffin_lim = T.GriffinLim(n_fft=hparams.filter_length,
                               win_length=hparams.win_length,
                               hop_length=hparams.hop_length,
                               power=2)

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
            # y_pred = model(x)
            y_pred = model.inference(x[0])
            print("Y_PRED[1] shape", y_pred.shape)  # Do I want y_pred[0] if batch size is 1?

            # Get denormalized signal for visualization and audio
            out_sig = dynamic_range_decompression(y_pred, .2)

            # Show or save Spectrogram
            save_spectrogram(out_sig, "Output/test/{i}_predicted.png")

            loss = criterion(y_pred, y)
            reduced_test_loss = loss.item()
            test_loss += reduced_test_loss

            # Get waveform from spectrogram
            powr_spec = power_spec(out_sig)  # This can handle y_pred and do the whole batch
            reconstructed_waveform = griffin_lim(powr_spec)

            # Save output audio
            torchaudio.save("Output/test/{i}_predicted.wav", reconstructed_waveform, hparams.sampling_rate)


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
