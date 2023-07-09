import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.optim as optim

def build(args, cfg):
    print("Building model: {}".format(args.model))
    if args.model == 'RegNet':
        from RegNet import RegNet
        model = RegNet.RegNet(args,cfg)
        return model