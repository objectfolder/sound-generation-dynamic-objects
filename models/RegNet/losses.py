import torch
from torch import nn

class RegnetLoss(nn.Module):
    def __init__(self, loss_type):
        super(RegnetLoss, self).__init__()
        self.loss_type = loss_type
        print("Loss type: {}".format(self.loss_type))

    def forward(self, model_output, targets):

        mel_target = targets
        mel_target.requires_grad = False
        mel_out, mel_out_postnet = model_output

        if self.loss_type == "MSE":
            loss_fn = nn.MSELoss()
        elif self.loss_type == "L1Loss":
            loss_fn = nn.L1Loss()
        else:
            print("ERROR LOSS TYPE!")

        mel_loss = loss_fn(mel_out, mel_target) + \
                   loss_fn(mel_out_postnet, mel_target)

        return mel_loss
    
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=False, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)