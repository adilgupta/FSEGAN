import torch.nn as nn
import numpy as np
import pandas as pd
import os
import torch
import glob
import scipy.io.wavfile as wavfile
from python_speech_features import logfbank, fbank
import pdb
def get_name(a):
    name = ""
    for temp in a:
        name = name + temp + "_"
    name = name[:-1]
    return name

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.filenames = glob.glob('/media/Sharedata/adil/timit/noisy_files/*.wav')
    def __getitem__(self, idx):
        rate, noisy_file = wavfile.read(self.filenames[idx])
        rate, clean_file = wavfile.read('/media/Sharedata/adil/timit/clean_files/' + get_name(self.filenames[idx].split('/')[-1].split('_')[:-2])+'.wav')
        if(len(clean_file) > 1*rate):
            start = np.random.randint(len(clean_file) - 1*rate)
            clean_file = clean_file[start:start+1*rate]
            noisy_file = noisy_file[start:start+1*rate]
        else:
            clean_file = np.concatenate((clean_file, np.zeros(1*rate-len(clean_file))), axis = 0)
            noisy_file = np.concatenate((noisy_file, np.zeros(1*rate-len(noisy_file))), axis = 0)
        feat_n, energy = fbank(noisy_file, samplerate=rate, nfilt=38, winfunc=np.hamming)
        feat_c, energy = fbank(clean_file, samplerate=rate, nfilt=38, winfunc=np.hamming)
        noisy_sfm = np.log(feat_n)
        clean_sfm = np.log(feat_c)
        return torch.from_numpy(noisy_sfm).type(torch.FloatTensor).view(1,noisy_sfm.shape[0], noisy_sfm.shape[1]), torch.from_numpy(clean_sfm).type(torch.FloatTensor).view(1,clean_sfm.shape[0], clean_sfm.shape[1])

    def __len__(self):
        return len(self.filenames)
