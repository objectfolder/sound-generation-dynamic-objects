import os
import os.path as osp
import sys
import json
from pprint import pprint

from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
import torch.optim as optim
import math
import librosa

from dataset.build import build as build_dataset
from models.build import build as build_model
from utils.wavenet_vocoder import builder
from utils.metrics import *
from utils.meters import AverageMeter
sys.path.append('../vocoders')
from waveglow.glow import WaveGlow

class Engine():
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        # set seeds
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        # build dataloaders
        self.train_loader, self.val_loader, self.test_loader = build_dataset(
            self.args, self.cfg)
        # build model & optimizer
        self.model = build_model(self.args, self.cfg)
        self.model.cuda()
        # experiment dir
        self.exp_dir = osp.join('./exp', self.args.exp)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.visualization=True
    
    def train_epoch(self, epoch):
        self.model.train()
        for i, batch in tqdm(enumerate(self.train_loader)):
            self.model.zero_grad()
            self.model.optimize_parameters(batch)
            loss = self.model.criterionL1((self.model.fake_B, self.model.fake_B_postnet), self.model.real_B)
            reduced_loss = loss.item()
            if i % 10 == 0:
                message = "epoch:{} iter:{} loss:{:.6f} G:{:.6f} D:{:.6f} D_r-f:{:.6f} G_s:{:.6f}".format(
                        epoch, i, 
                        reduced_loss, self.model.loss_G, 
                        self.model.loss_D, 
                        (self.model.pred_real - self.model.pred_fake).mean(), 
                        self.model.loss_G_silence)
                tqdm.write(message)
        tqdm.write(f"Finish Training Epoch {epoch}")
    
    def eval_epoch(self, epoch=0, test=False):
        self.model.eval()
        data_loader = self.test_loader if test else self.val_loader
        epoch_l2_distance = AverageMeter()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(data_loader)):
                self.model(batch)
                pred_spec = self.model.real_B.data.cpu().numpy()
                gt_spec = self.model.fake_B_postnet.data.cpu().numpy()
                l2_distance = STFT_L2_distance(pred_spec,gt_spec)
                epoch_l2_distance.update(l2_distance,pred_spec.shape[0])
                if self.visualization:
                    for j in range(len(self.model.fake_B)):
                        plt.figure(figsize=(8, 9))
                        plt.subplot(311)
                        plt.imshow(self.model.real_B[j].data.cpu().numpy(), 
                                        aspect='auto', origin='lower')
                        plt.title(self.model.video_name[j]+"_ground_truth")
                        plt.subplot(312)
                        plt.imshow(self.model.fake_B[j].data.cpu().numpy(), 
                                        aspect='auto', origin='lower')
                        plt.title(self.model.video_name[j]+"_predict")
                        plt.subplot(313)
                        plt.imshow(self.model.fake_B_postnet[j].data.cpu().numpy(), 
                                        aspect='auto', origin='lower')
                        plt.title(self.model.video_name[j]+"_postnet")
                        plt.tight_layout()
                        viz_dir = os.path.join(self.exp_dir, "viz", f'epoch_{epoch:05d}')
                        os.makedirs(viz_dir, exist_ok=True)
                        plt.savefig(os.path.join(viz_dir, self.model.video_name[j]+".jpg"))
                        plt.close()
                        spec_dir = os.path.join(self.exp_dir, "spectrogram", f'epoch_{epoch:05d}')
                        os.makedirs(spec_dir, exist_ok=True)
                        np.save(os.path.join(spec_dir, self.model.video_name[j]+".npy"), 
                                self.model.fake_B[j].data.cpu().numpy())
        return epoch_l2_distance.avg
    
    def train(self):
        bst_epoch_l2_distance = 1e8
        for epoch in range(self.args.epochs):
            print("Start Validation Epoch {}".format(epoch))
            epoch_l2_distance = self.eval_epoch(epoch)
            print("Finish Validation Epoch {}, L2 Distance = {:.8f}".format(epoch, epoch_l2_distance))
            if epoch_l2_distance < bst_epoch_l2_distance:
                print("New best L2 Distance {:.8f} reached, saving best model".format(epoch_l2_distance))
                bst_epoch_l2_distance = epoch_l2_distance
                torch.save(self.model.state_dict(), osp.join(self.exp_dir,'bst.pth'))
            torch.save(self.model.state_dict(), osp.join(self.exp_dir,'latest.pth'))
            print("Start Training Epoch {}".format(epoch))
            self.train_epoch(epoch)
        print("Finish Training Process")
        
    def load_waveglow(self, ckpt_path='../vocoders/exp/waveglow_small/ckpts/epoch_178.pth'):
        waveglow = WaveGlow(
            n_mel_channels=80,
            n_flows=12,
            n_group=6,
            n_early_every=4,
            n_early_size=2,
            WN_config={
                "n_layers": 4,
                "n_channels": 128,
                "kernel_size": 3
            }
        )
        state_dict = torch.load(ckpt_path)
        waveglow.load_state_dict(state_dict)
        return waveglow
    
    def evaluate_on_waveform(self):
        # evaluate on waveform
        waveglow = self.load_waveglow().cuda()
        test_envelope_distance = AverageMeter()
        test_cdpam_distance = AverageMeter()
        for root,dirs,files in os.walk(osp.join(self.exp_dir,'spectrogram','epoch_00000')):
            for file in tqdm(files):
                spec_path = osp.join(root,file)
                spec = np.load(spec_path)
                spec = torch.FloatTensor(spec).cuda()
                pred_wave = waveglow.infer(spec.unsqueeze(0), 1.0).cpu().numpy()[0]
                gt_wave,sr = librosa.load(osp.join(self.args.audio_location,'{}.wav'.format(spec_path.split('/')[-1].split('.')[0])))
                
                envelope_distance = Envelope_distance(pred_wave[:sr*5],gt_wave[:sr*5])
                cdpam_distance = CDPAM_distance(torch.tensor(pred_wave).unsqueeze(0).cuda(),
                               torch.tensor(gt_wave).unsqueeze(0).cuda())
                
                test_envelope_distance.update(envelope_distance)
                test_cdpam_distance.update(float(cdpam_distance))
        
        return test_envelope_distance.avg, test_cdpam_distance.avg
    
    def test(self):
        print("Start Testing")
        print("Loading best model from {}".format(osp.join(self.exp_dir,'bst.pth')))
        self.model.load_state_dict(torch.load(osp.join(self.exp_dir, 'bst.pth')))
        test_l2_distance = self.eval_epoch(test = True)
        test_envelope_distance, test_cdpam_distance = self.evaluate_on_waveform()
        result = {
            "Test Result":{
                'L2 Distance': test_l2_distance,
                'Envelope Distance': test_envelope_distance,
                'CDPAM Distance': test_cdpam_distance,
            }
        }
        json.dump(result, open(osp.join(self.exp_dir, 'result.json'),'w'))
        print("Finish Testing, L2 Distance = {:.8f}".format(test_l2_distance))
    
    def __call__(self):
        if not self.args.eval:
            self.train()
        self.test()
        