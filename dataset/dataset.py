import os
import pickle
import random
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class Video_Sound_Prediction_Dataset(Dataset):
    """
    loads image, flow feature, mel-spectrogramsfiles
    """
    def __init__(self, args, cfg, set_type = 'train'):
        self.args = args
        self.cfg = cfg
        self.set_type = set_type
        self.video_samples = self.cfg.video_samples
        self.audio_samples = self.cfg.audio_samples
        self.mel_samples = self.cfg.mel_samples
        with open(self.args.split_location) as f:
            video_list = json.load(f)[self.set_type] # [video_id]
            self.video_list = [i.strip() for i in video_list]
    
    def load_vision_feature(self, feature_path):
        with open(feature_path, 'rb') as f:
            feature = pickle.load(f, encoding='bytes')
        if feature.shape[0] < self.video_samples:
            feature_padded = np.zeros((self.video_samples, feature.shape[1]))
            feature_padded[0:feature.shape[0], :] = feature
        else:
            feature_padded = feature[0:self.video_samples, :]
        assert feature_padded.shape[0] == self.video_samples
        return feature_padded
    
    def load_mel(self, mel_path):
        melspec = np.load(mel_path)
        if melspec.shape[1] < self.mel_samples:
            melspec_padded = np.zeros((melspec.shape[0], self.mel_samples))
            melspec_padded[:, 0:melspec.shape[1]] = melspec
        else:
            melspec_padded = melspec[:, 0:self.mel_samples]
        return melspec_padded
    
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, index):
        data = {}
        video_id = self.video_list[index]
        rgb_feature_path = os.path.join(self.args.rgb_feature_location, video_id+".pkl")
        flow_feature_path = os.path.join(self.args.flow_feature_location, video_id+".pkl")
        mel_path = os.path.join(self.args.mel_location, video_id+"_mel.npy")
        rgb_feature = self.load_vision_feature(rgb_feature_path)
        flow_feature = self.load_vision_feature(flow_feature_path)
        mel = self.load_mel(mel_path)
        
        vision_feature = np.concatenate((rgb_feature, flow_feature), 1)
        vision_feature = torch.FloatTensor(vision_feature.astype(np.float32))
        mel = torch.FloatTensor(mel.astype(np.float32))
        data['video_id'] = video_id
        data['vision_feature'] = vision_feature
        data['mel'] = mel
        return data
    
    def collate(self, data):
        batch = {}
        for k in data[0].keys():
            if k =='video_id':
                batch[k] = [item[k] for item in data]
            else:
                batch[k] = torch.stack([item[k] for item in data])
        
        return batch
    
        