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
sys.path.append('../utils/vocoders')
sys.path.append('../utils/vocoders/melgan')
from melgan.mel2wav.modules import Generator
from utils.metrics import *
from utils.meters import AverageMeter
import soundfile
    
def load_melgan(ckpt_path='../utils/vocoders/melgan/logs/2022-11-09T06-28-09/best_netG.pt'):
    melgan = Generator(80, 32, 3).cuda()
    melgan.eval()
    state_dict = torch.load(ckpt_path)
    melgan.load_state_dict(state_dict)
    return melgan
    

gt_audio_path = '../DATA/features/audio_10s_22050hz'
gt_spec_path = '../DATA/features/melspec_10s_22050hz'
pred_spec_path = './exp/RegNet_1024/spectrogram/epoch_00000'

audio_save_path = './exp/RegNet_1024/audio_scaled'
sr=22050


melgan = load_melgan().cuda()

test_l2_distance = AverageMeter()
test_envelope_distance = AverageMeter()
test_cdpam_distance = AverageMeter()

for root, dirs, files in os.walk(pred_spec_path):
    for file in tqdm(files):
        if not '_9' in file:
            continue
        pred_spec = np.load(osp.join(root,file))[:,:430]
        gt_spec = np.load(osp.join(gt_spec_path,f"{file.split('.')[0]}_mel.npy"))[:,:430]
        
        with torch.no_grad():
            pred_audio = melgan(torch.tensor(pred_spec).cuda().unsqueeze(0)).cpu().numpy()[0][0]
            # pred_audio /= np.max(np.abs(pred_audio))
        gt_audio,_ = librosa.load(osp.join(gt_audio_path,f"{file.split('.')[0]}.wav"),sr=sr)
        gt_audio = gt_audio[:pred_audio.shape[0]]
        
        l2_dist = np.mean((pred_spec-gt_spec)**2)
        envelope_dist = Envelope_distance(pred_audio,gt_audio)
        cdpam_dist = CDPAM_distance(torch.tensor(pred_audio).unsqueeze(0).cuda(),
                                    torch.tensor(gt_audio).unsqueeze(0).cuda())
        
        test_l2_distance.update(float(l2_dist))
        test_envelope_distance.update(float(envelope_dist))
        test_cdpam_distance.update(float(cdpam_dist))
        
        pred_audio /= np.max(np.abs(pred_audio))
        soundfile.write(osp.join(audio_save_path,f"{file.split('.')[0]}.wav"),
                        np.concatenate([gt_audio,pred_audio]), sr)

print("test_l2_distance: {:.8f}".format(test_l2_distance.avg))
print("test_envelope_distance: {:.8f}".format(test_envelope_distance.avg))
print("test_cdpam_distance: {:.8f}".format(test_cdpam_distance.avg))