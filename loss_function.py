from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss


class Iso_Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Iso_Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        print("Gate target", gate_target)
        print("Gate target", gate_target.shape)
        # print("mel_target", mel_target)
        print("mel_target", mel_target.shape)
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)
        # print("Gate target view", gate_target)
        print("Gate target view", gate_target.shape)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        print("mel out", mel_out.shape)
        print("Mel out", mel_out)
        print("mel_out_postnet", mel_out_postnet.shape)
        print("gate out", gate_out)
        print("gate out", gate_out.shape)
        gate_out = gate_out.view(-1, 1)
        print("gate out view", gate_out.shape)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        print("Mel Loss", mel_loss)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        print("Gate loss", gate_loss)
        print()

        sample_rate = 22050
        hop_length = 256
        print(mel_target.shape[2])
        duration_target = mel_target.shape[2] / sample_rate * hop_length
        print(duration_target)
        duration_out = mel_out.shape[2] / sample_rate * hop_length
        print(duration_out)
        diff = abs(duration_target - duration_out)

        return mel_loss + gate_loss