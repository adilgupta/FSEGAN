import torch.nn as nn
import numpy as np
import os
import torch
import scipy.io.wavfile as wavfile
from operator import add
from python_speech_features import logfbank, fbank
from read_yaml import read_yaml
import pdb
import os


class G(nn.Module):
    def __init__(self):
        super().__init__()

        # input shape - B x 1 x frames x num_filters
        self.enc1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 4, stride = 1, padding = 0)
        self.enc1_nl = nn.PReLU()
        self.enc2 = nn.Conv2d(16, 32, 4, 1, 0)
        self.enc2_nl = nn.PReLU()
        self.enc3 = nn.Conv2d(32, 32, 4, 1, 0)
        self.enc3_nl = nn.PReLU()
        self.enc4 = nn.Conv2d(32, 64, 4, 1, 0)
        self.enc4_nl = nn.PReLU()
        self.enc5 = nn.Conv2d(64, 64, 4, 1, 0)
        self.enc5_nl = nn.PReLU()
        self.enc6 = nn.Conv2d(64, 128, 4, 1, 0)
        self.enc6_nl = nn.PReLU()
        self.enc7 = nn.Conv2d(128, 128, 4, 1, 0)
        self.enc7_nl = nn.PReLU()
        self.enc8 = nn.Conv2d(128, 256, 4, 1, 0)
        self.enc8_nl = nn.PReLU()
        self.enc9 = nn.Conv2d(256, 256, 4, 1, 0)
        self.enc9_nl = nn.PReLU()
        self.enc10 = nn.Conv2d(256, 512, 4, 1, 0)
        self.enc10_nl = nn.PReLU()
        self.enc11 = nn.Conv2d(512, 512, 4, 1, 0)
        self.enc11_nl = nn.PReLU()


        self.dec11 = nn.ConvTranspose2d(512, 512, 4, 1, 0)
        self.dec11_nl = nn.PReLU()
        self.dec10 = nn.ConvTranspose2d(1024, 256, 4, 1, 0)
        self.dec10_nl = nn.PReLU()
        self.dec9 = nn.ConvTranspose2d(512, 256, 4, 1, 0)
        self.dec9_nl = nn.PReLU()
        self.dec8 = nn.ConvTranspose2d(512, 128, 4, 1, 0)
        self.dec8_nl = nn.PReLU()
        self.dec7 = nn.ConvTranspose2d(256, 128, 4, 1, 0)
        self.dec7_nl = nn.PReLU()
        self.dec6 = nn.ConvTranspose2d(256, 64, 4, 1, 0)
        self.dec6_nl = nn.PReLU()
        self.dec5 = nn.ConvTranspose2d(128, 64, 4, 1, 0)
        self.dec5_nl = nn.PReLU()
        self.dec4 = nn.ConvTranspose2d(128, 32, 4, 1, 0)
        self.dec4_nl = nn.PReLU()
        self.dec3 = nn.ConvTranspose2d(64, 32, 4, 1, 0)
        self.dec3_nl = nn.PReLU()
        self.dec2 = nn.ConvTranspose2d(64, 16, 4, 1, 0)
        self.dec2_nl = nn.PReLU()
        self.dec1 = nn.ConvTranspose2d(in_channels = 32, out_channels = 1, kernel_size = 4, stride = 1, padding = 0)
        #self.dec_tanh = nn.Linear()

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal(m.weight.data)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.enc1_nl(e1))
        e3 = self.enc3(self.enc2_nl(e2))
        e4 = self.enc4(self.enc3_nl(e3))
        e5 = self.enc5(self.enc4_nl(e4))
        e6 = self.enc6(self.enc5_nl(e5))
        e7 = self.enc7(self.enc6_nl(e6))
        e8 = self.enc8(self.enc7_nl(e7))
        e9 = self.enc9(self.enc8_nl(e8))
        e10 = self.enc10(self.enc9_nl(e9))
        e11 = self.enc11(self.enc10_nl(e10))
        c = self.enc11_nl(e11)

        d11 = self.dec11(c)
        d11_c = self.dec11_nl(torch.cat((d11, e10), dim = 1))
        d10 = self.dec10(d11_c)
        d10_c = self.dec10_nl(torch.cat((d10, e9), dim = 1))
        d9 = self.dec9(d10_c)
        d9_c = self.dec9_nl(torch.cat((d9, e8), dim = 1))
        d8 = self.dec8(d9_c)
        d8_c = self.dec8_nl(torch.cat((d8, e7), dim = 1))
        d7 = self.dec7(d8_c)
        d7_c = self.dec7_nl(torch.cat((d7, e6), dim = 1))
        d6 = self.dec6(d7_c)
        d6_c = self.dec6_nl(torch.cat((d6, e5), dim = 1))
        d5 = self.dec5(d6_c)
        d5_c = self.dec5_nl(torch.cat((d5, e4), dim = 1))
        d4 = self.dec4(d5_c)
        d4_c = self.dec4_nl(torch.cat((d4, e3), dim = 1))
        d3 = self.dec3(d4_c)
        d3_c = self.dec3_nl(torch.cat((d3, e2), dim = 1))
        d2 = self.dec2(d3_c)
        d2_c = self.dec2_nl(torch.cat((d2, e1), dim = 1))
        d1 = self.dec1(d2_c)
        #out = self.dec_tanh(d1)

        return d1#out

class D(nn.Module):
    """D"""

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
        nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 4, stride = 1, padding = 0),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(16))
        self.block2 = nn.Sequential(
        nn.Conv2d(16, 32, 4, 1, 0),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(32))
        self.block3 = nn.Sequential(
        nn.Conv2d(32, 32, 4, 1, 0),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(32))
        self.block4 = nn.Sequential(
        nn.Conv2d(32, 64, 4, 1, 0),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(64))
        self.block5 = nn.Sequential(
        nn.Conv2d(64, 64, 4, 1, 0),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(64))
        self.block6 = nn.Sequential(
        nn.Conv2d(64, 128, 4, 1, 0),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(128))
        self.block7 = nn.Sequential(
        nn.Conv2d(128, 128, 4, 1, 0),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(128))
        self.block8 = nn.Sequential(
        nn.Conv2d(128, 256, 4, 1, 0),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(256))
        self.block9 = nn.Sequential(
        nn.Conv2d(256, 256, 4, 1, 0),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(256))
        self.block10 = nn.Sequential(
        nn.Conv2d(256, 512, 4, 1, 0),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(512))
        self.block11 = nn.Sequential(
        nn.Conv2d(512, 512, 4, 1, 0),
        nn.PReLU())

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal(m.weight.data)

    def forward(self, x):
        batchsize = x.size()[0]
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out5 = self.block5(out4)
        out6 = self.block6(out5)
        out7 = self.block7(out6)
        out8 = self.block8(out7)
        out9 = self.block9(out8)
        out10 = self.block10(out9)
        out11 = self.block11(out10)
        output = torch.cat((x.view(batchsize,-1),out1.view(batchsize,-1),
                                out2.view(batchsize,-1),out3.view(batchsize,-1),
                                out4.view(batchsize,-1),out5.view(batchsize,-1),
                                out6.view(batchsize,-1), out7.view(batchsize,-1),
                                out8.view(batchsize,-1),out9.view(batchsize,-1),
                                out10.view(batchsize,-1),out11.view(batchsize,-1) ),1)
        return output # out11.view(batchsize, -1)#output

class dl_model():

    def __init__(self, mode):

        # read config fiel which contains parameters
        self.config_file = read_yaml()
        self.mode = mode

        arch_name = '_'.join(
            [self.config_file['rnn'], str(self.config_file['num_layers']), str(self.config_file['hidden_dim'])])
        self.config_file['dir']['models'] = self.config_file['dir']['models'].split('/')[0] + '_' + arch_name + '/'
        self.config_file['dir']['plots'] = self.config_file['dir']['plots'].split('/')[0] + '_' + arch_name + '/'

        #if not os.path.exists(self.config_file['dir']['models']):
        #    os.mkdir(self.config_file['dir']['models'])
        #if not os.path.exists(self.config_file['dir']['plots']):
        #    os.mkdir(self.config_file['dir']['plots'])

        if self.config_file['rnn'] == 'LSTM':
            from model import LSTM as Model
        elif self.config_file['rnn'] == 'GRU':
            from model import GRU as Model
        else:
            print("Model not implemented")
            exit(0)

        self.cuda = (self.config_file['cuda'] and torch.cuda.is_available())
        self.output_dim = self.config_file['num_phones']

        if mode == 'train' or mode == 'test':

            self.plots_dir = self.config_file['dir']['plots']
            # store hyperparameters
            self.total_epochs = self.config_file['train']['epochs']
            self.test_every = self.config_file['train']['test_every_epoch']
            self.test_per = self.config_file['train']['test_per_epoch']
            self.print_per = self.config_file['train']['print_per_epoch']
            self.save_every = self.config_file['train']['save_every']
            self.plot_every = self.config_file['train']['plot_every']
            # dataloader which returns batches of data
            self.train_loader = timit_loader('train', self.config_file)
            self.test_loader = timit_loader('test', self.config_file)

            self.start_epoch = 1
            self.test_acc = []
            self.train_losses, self.test_losses = [], []
            # declare model
            self.model = Model(self.config_file, weights=self.train_loader.weights)

        else:

            self.model = Model(self.config_file, weights=None)

        if self.cuda:
            self.model.cuda()

        # resume training from some stored model
        if self.mode == 'train' and self.config_file['train']['resume']:
            self.start_epoch, self.train_losses, self.test_losses, self.test_acc = self.model.load_model(mode,
                                                                                                         self.config_file[
                                                                                                             'rnn'],
                                                                                                         self.model.num_layers,
                                                                                                         self.model.hidden_dim)
            self.start_epoch += 1

        # load best model for testing/feature extraction
        elif self.mode == 'test' or mode == 'test_one':
            self.model.load_model(mode, self.config_file['rnn'], self.model.num_layers, self.model.hidden_dim)

        self.replacement = {'aa': ['ao'], 'ah': ['ax', 'ax-h'], 'er': ['axr'], 'hh': ['hv'], 'ih': ['ix'],
                            'l': ['el'], 'm': ['em'], 'n': ['en', 'nx'], 'ng': ['eng'], 'sh': ['zh'],
                            'pau': ['pcl', 'tcl', 'kcl', 'bcl', 'dcl', 'gcl', 'h#', 'epi', 'q'],
                            'uw': ['ux']}

def generate_lattice(outputs, h_spike):
    tsteps, num_phones = outputs.shape
    lattice = [[] for i in range(tsteps)]
    for i in range(tsteps):
        for j in range(num_phones):
            if outputs[i][j] >= h_spike:
                lattice[i].append((j, outputs[i][j]))

    # Collapse consecutive
    final_lattice = []

    previous_phones = [x[0] for x in lattice[0]]
    prev_sum = [x[1] for x in lattice[0]]
    num = 1

    for l in lattice[1:]:
        ids, vals = [x[0] for x in l], [x[1] for x in l]

        if ids == previous_phones:
            num += 1
            prev_sum = list(map(add, prev_sum, vals))
        else:
            final_lattice.append(list(zip(previous_phones, [x / num for x in prev_sum])))
            previous_phones = ids
            prev_sum = vals
            num = 1

    final_lattice.append(list(zip(previous_phones, [x / num for x in prev_sum])))

    return final_lattice


if __name__=="__main__":
    rate, sig = wavfile.read('./SA1.WAV.wav')
    feat, energy = fbank(sig, samplerate=rate, nfilt=38, winfunc=np.hamming)
    #feat = np.log(feat)
    tsteps, hidden_dim = feat.shape
    feat_log_full = np.reshape(np.log(feat), (1, tsteps, hidden_dim))
    lens = np.array([tsteps])
    inputs, lens = torch.from_numpy(np.array(feat_log_full)).float(), torch.from_numpy(np.array(lens)).long()
    dl_model = dl_model("test_one")
    id_to_phone = {v[0]: k for k, v in dl_model.model.phone_to_id.items()}

    dl_model.model.eval()
    with torch.no_grad():
        #if cuda:
        #    inputs = inputs.cuda()
        #    lens = lens.cuda()

        # Pass through model
        outputs = dl_model.model(inputs, lens).cpu().numpy()
        # Since only one example per batch and ignore blank token
        outputs = outputs[0, :, :-1]
        softmax = np.exp(outputs) / np.sum(np.exp(outputs), axis=1)[:, None]

    outputs, mapping = softmax, id_to_phone
    final_lattice = generate_lattice(outputs, 0.2)
    print(final_lattice, ([len(x) for x in final_lattice if len(x) != 1]))

    phones = [[mapping[x[0]] for x in l] for l in final_lattice]
    print(phones)
    pdb.set_trace()

    """

    gen = G()
    dis = D()
    out = gen(torch.tensor(feat, dtype = torch.float32).view(1,1,feat.shape[0],feat.shape[1]))
    """
    #phseq = dl_model('')
    pdb.set_trace()
