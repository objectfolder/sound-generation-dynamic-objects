import sys
sys.path.insert(0, '.')  # nopep8

from mel2wav.dataset import AudioDataset
from mel2wav.modules import Generator, Discriminator
from mel2wav.utils import save_sample, wav2mel

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True

import yaml
import numpy as np
import time
import argparse
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--load_path", default=None)

    parser.add_argument("--n_mel_channels", type=int, default=80)
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=3)

    parser.add_argument("--ndf", type=int, default=16)
    parser.add_argument("--num_D", type=int, default=3)
    parser.add_argument("--n_layers_D", type=int, default=4)
    parser.add_argument("--downsamp_factor", type=int, default=4)
    parser.add_argument("--lambda_feat", type=float, default=10)
    parser.add_argument("--cond_disc", action="store_true")
    
    parser.add_argument("--data_path", default=None, type=Path)
    parser.add_argument("--splits_path", default='./data', type=Path)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=8192)

    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--n_test_samples", type=int, default=16)
    args = parser.parse_args()
    return args

def build_model(args):
    print('netG init...')
    netG = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).cuda()
    print('loading vggsound pretrain...')
    netG.load_state_dict(torch.load("./logs/vggsound/best_netG.pt"))
    print('netD init...')
    netD = Discriminator(args.num_D, args.ndf, args.n_layers_D, args.downsamp_factor).cuda()
    netD.load_state_dict(torch.load("./logs/2022-11-09T06-03-47/netD.pt"))
    print('optimizer init...')
    optG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.9))
    optD = torch.optim.Adam(netD.parameters(), lr=args.lr*10, betas=(0.5, 0.9))
    
    # optG = torch.optim.Adam(netG.parameters(), lr=0, betas=(0.5, 0.9))
    # optD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    
    return netG, netD, optG, optD

def build_dataset(args):
    print('dataset init...')
    train_set = AudioDataset(
        Path(args.data_path), Path(args.splits_path) / 'objectfolder_train.txt', args.seq_len, sampling_rate=22050
    )
    valid_set = AudioDataset(
        Path(args.data_path),
        Path(args.splits_path) / 'objectfolder_valid.txt',
        220160,
        sampling_rate=22050,
        augment=False,
    )
    test_set = AudioDataset(
        Path(args.data_path),
        Path(args.splits_path) / 'objectfolder_test.txt',
        22050 * 10,
        sampling_rate=22050,
        augment=False,
    )
    print("Dataset Loaded, train: {}, valid: {}".format(len(train_set)//args.batch_size,len(valid_set)))
    print('dataloader init...')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=1, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=1)
    return train_loader, valid_loader, test_loader

def train_epoch(args, epoch, train_loader, netG, netD, optG, optD):
    costs = []
    netG.train()
    netD.train()
    for iterno, x_t in tqdm(enumerate(train_loader)):
        x_t = x_t.cuda()
        trim_len = x_t.shape[-1] // 256
        s_t = wav2mel(x_t.squeeze(1), trim_len)
        x_pred_t = netG(s_t.cuda())

        with torch.no_grad():
            s_pred_t = wav2mel(x_pred_t.squeeze(1).detach().cpu(), trim_len)
            s_error = F.l1_loss(s_t, s_pred_t).item()
        
        D_fake_det = netD(x_pred_t.detach())
        D_real = netD(x_t)

        loss_D = 0
        for scale in D_fake_det:
            loss_D += F.relu(1 + scale[-1]).mean()

        for scale in D_real:
            loss_D += F.relu(1 - scale[-1]).mean()

        netD.zero_grad()
        loss_D.backward()
        optD.step()
            
        D_fake = netD(x_pred_t)

        loss_G = 0
        for scale in D_fake:
            loss_G += -scale[-1].mean()

        loss_feat = 0
        feat_weights = 4.0 / (args.n_layers_D + 1)
        D_weights = 1.0 / args.num_D
        wt = D_weights * feat_weights
        for i in range(args.num_D):
            for j in range(len(D_fake[i]) - 1):
                loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())

        # netG.zero_grad()
        # (loss_G + args.lambda_feat * loss_feat).backward()
        # optG.step()
        
        costs.append([loss_D.item(), loss_G.item(), loss_feat.item(), s_error])
    print(
            "Training Epoch {} | loss {}".format(
                epoch,
                np.asarray(costs).mean(0)
                )
        )
    return costs

def valid_epoch(args, epoch, valid_loader, netG, netD):
    costs = []
    netG.eval()
    netD.eval()
    for iterno, x_t in tqdm(enumerate(valid_loader)):
        x_t = x_t.cuda()
        trim_len = x_t.shape[-1] // 256
        s_t = wav2mel(x_t.squeeze(1), trim_len)
        x_pred_t = netG(s_t.cuda())

        with torch.no_grad():
            s_pred_t = wav2mel(x_pred_t.squeeze(1).detach().cpu(), trim_len)
            s_error = F.l1_loss(s_t, s_pred_t).item()
        
        D_fake_det = netD(x_pred_t.detach())
        D_real = netD(x_t)

        loss_D = 0
        for scale in D_fake_det:
            loss_D += F.relu(1 + scale[-1]).mean()

        for scale in D_real:
            loss_D += F.relu(1 - scale[-1]).mean()

        D_fake = netD(x_pred_t)

        loss_G = 0
        for scale in D_fake:
            loss_G += -scale[-1].mean()

        loss_feat = 0
        feat_weights = 4.0 / (args.n_layers_D + 1)
        D_weights = 1.0 / args.num_D
        wt = D_weights * feat_weights
        for i in range(args.num_D):
            for j in range(len(D_fake[i]) - 1):
                loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())

        costs.append([loss_D.item(), loss_G.item(), loss_feat.item(), s_error])
    print(
            "Validation Epoch {} | loss {}".format(
                epoch,
                np.asarray(costs).mean(0)
                )
        )
    return costs


def main():
    args = parse_args()
    
    root = Path(args.save_path)
    load_root = Path(args.load_path) if args.load_path else None
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)
    
    netG, netD, optG, optD = build_model(args)
    if load_root and load_root.exists():
        netG.load_state_dict(torch.load(load_root / "netG.pt"))
        optG.load_state_dict(torch.load(load_root / "optG.pt"))
        netD.load_state_dict(torch.load(load_root / "netD.pt"))
        optD.load_state_dict(torch.load(load_root / "optD.pt"))
        
    train_loader, valid_loader, test_loader = build_dataset(args)
    test_voc = []
    test_audio = []
    for i, x_t in tqdm(enumerate(test_loader)):
        x_t = x_t.cuda()
        s_t = wav2mel(x_t)

        test_voc.append(s_t.cuda())
        test_audio.append(x_t)

        audio = x_t.squeeze().cpu()
        save_sample(root / f'original_{i}.wav', 22050, audio)

        if i == args.n_test_samples - 1:
            break
    
    best_mel_reconst = 1000000
    for epoch in range(1, args.epochs + 1):
        valid_costs = valid_epoch(args, epoch, valid_loader,
                    netG, netD)
        if np.asarray(valid_costs).mean(0)[-1] < best_mel_reconst:
            print('saving best ckpt')
            best_mel_reconst = np.asarray(valid_costs).mean(0)[-1]
            torch.save(netD.state_dict(), root / "best_netD.pt")
            torch.save(netG.state_dict(), root / "best_netG.pt")
            print('testing...')
            netG.eval()
            with torch.no_grad():
                for i, (voc, _) in enumerate(zip(test_voc, test_audio)):
                    pred_audio = netG(voc)
                    pred_audio = pred_audio.squeeze().cpu()
                    save_sample(root / ("generated_%d.wav" % i), 22050, pred_audio)
        print('saving latest ckpt...')
        torch.save(netG.state_dict(), root / "netG.pt")
        torch.save(optG.state_dict(), root / "optG.pt")

        torch.save(netD.state_dict(), root / "netD.pt")
        torch.save(optD.state_dict(), root / "optD.pt")
        train_costs = train_epoch(args, epoch, train_loader,
                    netG, netD,
                    optG, optD)
        

if __name__ == "__main__":
    main()