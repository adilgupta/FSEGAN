import glob
import numpy as np
import torch
import torch.nn as nn
import scipy.io.wavfile as wavfile
from python_speech_features import logfbank, fbank
from read_yaml import read_yaml
from networks import *
import pdb
import os

num_files_to_test = 100

def get_name(a):
    name = ""
    for temp in a:
        name = name + temp + "_"
    name = name[:-1]
    return name

def get_phones_td(sig, rate, dl_model):
    feat, energy = fbank(sig, samplerate=rate, nfilt=38, winfunc=np.hamming)
    #feat = np.log(feat)
    tsteps, hidden_dim = feat.shape
    feat_log_full = np.reshape(np.log(feat), (1, tsteps, hidden_dim))
    lens = np.array([tsteps])
    inputs, lens = torch.from_numpy(np.array(feat_log_full)).float(), torch.from_numpy(np.array(lens)).long()
    id_to_phone = {v[0]: k for k, v in dl_model.model.phone_to_id.items()}
    with torch.no_grad():
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            lens = lens.cuda()
        # Pass through model
        outputs = dl_model.model(inputs, lens).cpu().numpy()
        # Since only one example per batch and ignore blank token
        outputs = outputs[0, :, :-1]
        softmax = np.exp(outputs) / np.sum(np.exp(outputs), axis=1)[:, None]

    outputs, mapping = softmax, id_to_phone
    final_lattice = generate_lattice(outputs, 0.2)
    #db.set_trace()
    #print(np.argmax(outputs, axis=1))
    phones = [[mapping[x[0]] for x in l] for l in final_lattice]
    return np.argmax(outputs, axis = 1)# phones

def get_phones_feat_map(feat, dl_model):
    tsteps, hidden_dim = feat.shape
    feat_log_full = np.reshape(feat, (1, tsteps, hidden_dim))
    lens = np.array([tsteps])
    inputs, lens = torch.from_numpy(np.array(feat_log_full)).float(), torch.from_numpy(np.array(lens)).long()
    id_to_phone = {v[0]: k for k, v in dl_model.model.phone_to_id.items()}
    with torch.no_grad():
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            lens = lens.cuda()
        # Pass through model
        outputs = dl_model.model(inputs, lens).cpu().numpy()
        # Since only one example per batch and ignore blank token
        outputs = outputs[0, :, :-1]
        softmax = np.exp(outputs) / np.sum(np.exp(outputs), axis=1)[:, None]

    outputs, mapping = softmax, id_to_phone
    final_lattice = generate_lattice(outputs, 0.2)
    #print(np.argmax(outputs, axis=1))
    phones = [[mapping[x[0]] for x in l] for l in final_lattice]
    return np.argmax(outputs, axis = 1)#phones

if __name__ == "__main__":
    noisy_files = glob.glob('/media/Sharedata/adil/timit/noisy_files/*.wav')
    gen = G()
    dl_model = dl_model("test_one")
    dl_model.model.eval()

    if torch.cuda.is_available():
        gen = nn.DataParallel(gen.cuda())
    gen.load_state_dict(torch.load('/media/Sharedata/adil/timit/epochs_adv_2/gen_epoch_142.pth'))
    gen.eval()
    clean_files = glob.glob('/media/Sharedata/adil/timit/clean_files/*.wav')
    for i in range(num_files_to_test):
        # noisy_file_name = DR4_MJWS0_SX153.WAV_rain_14.5db.wav
        # clean_file_name = DR4_MJRH0_SX135.WAV.wav
        rate, noisy_sig = wavfile.read(noisy_files[i])
        feat_n, energy = fbank(noisy_sig, samplerate=rate, nfilt=38, winfunc=np.hamming)
        noisy_sfm = np.log(feat_n)
        feat = torch.from_numpy(noisy_sfm).type(torch.FloatTensor).view(1, 1, noisy_sfm.shape[0], noisy_sfm.shape[1])
        with torch.no_grad():
            enhan_feat = gen(feat).cpu()
        enhan_feat = enhan_feat[0,0,:,:]
        #pdb.set_trace()
        rate, clean_sig = wavfile.read('/media/Sharedata/adil/timit/clean_files/' + get_name(noisy_files[i].split('/')[-1].split('_')[:-2])+'.wav')
        #rate, sig = wavfile.read(clean_files[i])


        phones_noisy = get_phones_td(noisy_sig, rate, dl_model)
        phones_clean = get_phones_td(clean_sig, rate, dl_model)
        phones_enhanced = get_phones_feat_map(enhan_feat, dl_model)
        #print(phones_clean)
        #print(phones_noisy)
        #print(phones_enhanced)
        pdb.set_trace()
