import os
import librosa
import argparse
import numpy as np
import torch
from numpy import linalg as LA
from cdpam import CDPAM
from scipy.signal import hilbert


def STFT_L2_distance(pred_spec, gt_spec):
    L2_distance = np.mean((gt_spec - pred_spec)**2)
    return L2_distance


def Envelope_distance(pred_signal, gt_signal):
    pred_env = np.abs(hilbert(pred_signal))
    gt_env = np.abs(hilbert(gt_signal))
    envelope_distance = np.sqrt(np.mean((gt_env - pred_env)**2))
    return float(envelope_distance)


def CDPAM_distance(pred_signal, gt_signal):
    loss_fn = CDPAM()
    with torch.no_grad():
        cdpam_distance = loss_fn.forward(pred_signal, gt_signal)
    return cdpam_distance
