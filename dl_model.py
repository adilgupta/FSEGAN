"""
The main driver file responsible for training, testing and extracting features
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import time
from read_yaml import read_yaml
from dataloader import timit_loader
import scipy.io.wavfile as wav
from python_speech_features import fbank


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

    def train(self):

        print("Starting training at t =", datetime.datetime.now())
        print('Batches per epoch:', len(self.train_loader))
        self.model.train()

        # when to print losses during the epoch
        print_range = list(np.linspace(0, len(self.train_loader), self.print_per + 2, dtype=np.uint32)[1:-1])

        if self.test_per == 0:
            test_range = []
        else:
            test_range = list(np.linspace(0, len(self.train_loader), self.test_per + 2, dtype=np.uint32)[1:-1])

        for epoch in range(self.start_epoch, self.total_epochs + 1):

            print("Epoch:", str(epoch))
            epoch_loss = 0.0
            i = 0

            while True:

                i += 1

                inputs, labels, lens, status = self.train_loader.return_batch()
                inputs, labels, lens = torch.from_numpy(np.array(inputs)).float(), torch.from_numpy(
                    np.array(labels)).long(), torch.from_numpy(np.array(lens)).long()

                if self.cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    lens = lens.cuda()

                # zero the parameter gradients
                self.model.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs, lens)
                loss = self.model.calculate_loss(outputs, labels, lens)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config_file['grad_clip'])
                self.model.optimizer.step()

                # store loss
                epoch_loss += loss.item()

                if i in print_range:
                    try:
                        print('After %i batches, Current Loss = %.7f, Avg. Loss = %.7f' % (
                            i + 1, epoch_loss / (i + 1), np.mean([x[0] for x in self.train_losses])))
                    except:
                        pass

                if i in test_range:
                    self.test(epoch)
                    self.model.train()

                if status == 1:
                    break

            self.train_losses.append((epoch_loss / len(self.train_loader), epoch))

            # test every 5 epochs in the beginning and then every fixed no of epochs specified in config file
            # useful to see how loss stabilises in the beginning
            if epoch % 5 == 0 and epoch < self.test_every:
                self.test(epoch)
                self.model.train()
            elif epoch % self.test_every == 0:
                self.test(epoch)
                self.model.train()
            # plot loss and accuracy
            if epoch % self.plot_every == 0:
                self.plot_loss_acc(epoch)

            # save model
            if epoch % self.save_every == 0:
                self.model.save_model(False, epoch, self.train_losses, self.test_losses, self.test_acc,
                                      self.config_file['rnn'], self.model.num_layers, self.model.hidden_dim)

    def test(self, epoch=None):

        self.model.eval()
        correct = 0
        total = 0
        correct_nopause = 0
        total_nopause = 0
        pause_id = 27
        # confusion matrix data is stored in this matrix
        matrix = np.zeros((self.output_dim, self.output_dim))
        pad_id = self.output_dim

        print("Testing...")
        print('Total batches:', len(self.test_loader))
        test_loss = 0

        with torch.no_grad():

            while True:

                inputs, labels, lens, status = self.train_loader.return_batch()
                inputs, labels, lens = torch.from_numpy(np.array(inputs)).float(), torch.from_numpy(
                    np.array(labels)).long(), torch.from_numpy(np.array(lens)).long()
                # print(inputs.shape, labels.shape, lens)
                if self.cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    lens = lens.cuda()

                # zero the parameter gradients
                self.model.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs, lens)
                loss = self.model.calculate_loss(outputs, labels, lens)
                test_loss += loss.item()

                outputs = outputs.cpu().numpy()
                labels = labels.cpu().numpy()[:, :outputs.shape[1]]  # remove extra padding from current batch
                outputs = np.reshape(outputs[:, :, :-1], (-1, self.output_dim))  # ignore blank token
                labels = np.reshape(labels, (-1))
                total_pad_tokens = np.sum(labels == pad_id)
                argmaxed = np.argmax(outputs, 1)

                # total number of correct phone predictions
                for i in range(len(labels)):
                    if labels[i] != pause_id and labels[i] != pad_id:  # is not pause or pad
                        if argmaxed[i] == labels[i]:
                            correct_nopause += 1
                        total_nopause += 1
                correct += np.sum(argmaxed == labels)
                total += len(argmaxed) - total_pad_tokens

                # matrix[i][j] denotes the no of examples classified by model as class j but have ground truth label i
                for k in range(argmaxed.shape[0]):
                    if labels[k] == pad_id:
                        continue
                    matrix[labels[k]][argmaxed[k]] += 1

                if status == 1:
                    break

        for i in range(self.output_dim):
            matrix[i] /= sum(matrix[i])

        acc_all = correct / total
        acc_nopause = correct_nopause / total_nopause
        print(acc_all, acc_nopause)

        test_loss /= len(self.test_loader)

        # plot confusion matrix
        if epoch is not None:
            filename = self.plots_dir + 'confmat_epoch_acc_' + str(epoch) + '_' + str(int(100 * acc_all)) + '.png'
            plt.clf()
            plt.imshow(matrix, cmap='hot', interpolation='none')
            plt.gca().invert_yaxis()
            plt.xlabel("Predicted Label ID")
            plt.ylabel("True Label ID")
            plt.colorbar()
            plt.savefig(filename)

        print("Testing accuracy: All - %.4f, No Pause - %.4f , Loss: %.7f" % (acc_all, acc_nopause, test_loss))

        self.test_acc.append((acc_all, epoch))
        self.test_losses.append((test_loss, epoch))

        # if testing loss is minimum, store it as the 'best.pth' model, which is used for feature extraction
        if test_loss == min([x[0] for x in self.test_losses]):
            print("Best new model found!")
            self.model.save_model(True, epoch, self.train_losses, self.test_losses, self.test_acc,
                                  self.config_file['rnn'], self.model.num_layers, self.model.hidden_dim)

        return acc_all

    # Called during feature extraction. Takes log mel filterbank energies as input and outputs the phone predictions
    def test_one(self, file_path):

        (rate, sig) = wav.read(file_path)
        assert rate == 16000
        # sig ranges from -32768 to +32768 AND NOT -1 to +1
        feat, energy = fbank(sig, samplerate=rate, nfilt=self.config_file['feat_dim'], winfunc=np.hamming)
        tsteps, hidden_dim = feat.shape
        # calculate log mel filterbank energies for complete file
        feat_log_full = np.reshape(np.log(feat), (1, tsteps, hidden_dim))
        lens = np.array([tsteps])
        inputs, lens = torch.from_numpy(np.array(feat_log_full)).float(), torch.from_numpy(np.array(lens)).long()
        id_to_phone = {v[0]: k for k, v in self.model.phone_to_id.items()}

        self.model.eval()

        with torch.no_grad():
            if self.cuda:
                inputs = inputs.cuda()
                lens = lens.cuda()

            # Pass through model
            a = time.time()

            outputs = self.model(inputs, lens).cpu().numpy()
            print(time.time()-a)
            # Since only one example per batch and ignore blank token
            outputs = outputs[0, :, :-1]
            softmax = np.exp(outputs) / np.sum(np.exp(outputs), axis=1)[:, None]

        return softmax, id_to_phone

    # Test for each wav file in the folder and also compare with ground truth
    def test_folder(self, test_folder, top_n=1, show_graphs=False):

        accs = []

        for wav_file in sorted(os.listdir(test_folder)):

            # Read input test file
            wav_path = os.path.join(test_folder, wav_file)
            dump_path = wav_path[:-4] + '_pred.txt'

            # Read only wav
            if wav_file == '.DS_Store' or wav_file.split('.')[-1] != 'wav':  # or os.path.exists(dump_path):
                continue

            (rate, sig) = wav.read(wav_path)
            assert rate == 16000
            # sig ranges from -32768 to +32768 AND NOT -1 to +1
            feat, energy = fbank(sig, samplerate=rate, nfilt=self.config_file['feat_dim'], winfunc=np.hamming)
            tsteps, hidden_dim = feat.shape
            # calculate log mel filterbank energies for complete file
            feat_log_full = np.reshape(np.log(feat), (1, tsteps, hidden_dim))
            lens = np.array([tsteps])
            inputs, lens = torch.from_numpy(np.array(feat_log_full)).float(), torch.from_numpy(np.array(lens)).long()
            id_to_phone = {v[0]: k for k, v in self.model.phone_to_id.items()}

            self.model.eval()

            with torch.no_grad():

                if self.cuda:
                    inputs = inputs.cuda()
                    lens = lens.cuda()

                # Pass through model
                outputs = self.model(inputs, lens).cpu().numpy()
                # Since only one example per batch and ignore blank token
                outputs = outputs[0, :, :-1]
                softmax = np.exp(outputs) / np.sum(np.exp(outputs), axis=1)[:, None]
                softmax_probs = np.max(softmax, axis=1)
                # print(softmax)
                # Take argmax ot generate final string
                argmaxed = np.argmax(outputs, axis=1)
                final_str = [id_to_phone[a] for a in argmaxed]
                # Generate dumpable format of phone, start time and end time
                ans = compress_seq(final_str)
                print("Predicted:", ans)

            phone_path = wav_path[:-3] + 'PHN'

            # If .PHN file exists, report accuracy
            if os.path.exists(phone_path):
                grtuth = read_phones(phone_path, self.replacement)
                print("Ground truth:", grtuth)

                unrolled_truth = []
                for elem in grtuth:
                    unrolled_truth += [elem[0]] * (elem[2] - elem[1] + 1)

                truth_softmax = []
                top_n_softmax = [[] for x in range(top_n)]
                # Check for top-n
                correct, total = 0, 0
                for i in range(min(len(unrolled_truth), len(final_str))):

                    truth_softmax.append(softmax[i][self.model.phone_to_id[unrolled_truth[i]][0]])

                    indices = list(range(len(final_str)))
                    zipped = zip(indices, outputs[i])
                    desc = sorted(zipped, key=lambda x: x[1], reverse=True)
                    cur_frame_res = [id_to_phone[x[0]] for x in desc][:top_n]

                    for k in range(top_n):
                        top_n_softmax[k].append(softmax[i][self.model.phone_to_id[cur_frame_res[k]][0]])

                    if unrolled_truth[i] in cur_frame_res:
                        # print truth softmax
                        # if unrolled_truth[i] != cur_frame_res[0]:
                        # print(i, truth_softmax[-1])
                        correct += 1

                    total += 1

                accs.append(correct / total)

                if show_graphs:
                    # Plot actual softmax and predicted softmax
                    for i in range(top_n):
                        plt.plot(top_n_softmax[i], label=str(i + 1) + ' prob.')
                    print(top_n_softmax)
                    plt.plot(truth_softmax, label='Ground Truth prob', alpha=0.6)
                    plt.xlabel("Frame number")
                    plt.ylabel("Prob")
                    plt.legend()
                    plt.show()

                with open(dump_path, 'w') as f:
                    f.write('Predicted:\n')
                    for t in ans:
                        f.write(' '.join(str(s) for s in t) + '\n')
                    f.write('\nGround Truth:\n')
                    for t in grtuth:
                        f.write(' '.join(str(s) for s in t) + '\n')
                    f.write('\nTop-' + str(top_n) + ' accuracy is ' + str(correct / total))
            else:
                with open(dump_path, 'w') as f:
                    f.write('Predicted:\n')
                    for t in ans:
                        f.write(' '.join(str(s) for s in t) + '\n')
        print(accs)

    # take train/test loss and test accuracy input and plot it over time
    def plot_loss_acc(self, epoch):

        plt.clf()
        plt.plot([x[1] for x in self.train_losses], [x[0] for x in self.train_losses], c='r', label='Train')
        plt.plot([x[1] for x in self.test_losses], [x[0] for x in self.test_losses], c='b', label='Test')
        plt.title("Train/Test loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        filename = self.plots_dir + 'loss' + '_' + str(epoch) + '.png'
        plt.savefig(filename)

        plt.clf()
        plt.plot([x[1] for x in self.test_acc], [100 * x[0] for x in self.test_acc], c='r')
        plt.title("Test accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy in %%")
        plt.grid(True)

        filename = self.plots_dir + 'test_acc' + '_' + str(epoch) + '.png'
        plt.savefig(filename)

        print("Saved plots")


def compress_seq(data):
    final = []
    current_ph, current_start_idx = data[0], 0

    for i in range(2, len(data)):
        now_ph = data[i]
        if now_ph == current_ph:
            continue
        else:
            final.append((current_ph, current_start_idx, i - 1))
            current_start_idx = i
            current_ph = now_ph
    final.append((current_ph, current_start_idx, len(data) - 1))
    return final


def read_phones(phone_file_path, replacement):
    labels = []

    with open(phone_file_path, 'r') as f:
        a = f.readlines()

    for phone in a:
        s_e_i = phone[:-1].split(' ')  # start, end, phenome_name e.g. 0 5432 'aa'
        start, end, ph = int(s_e_i[0]), int(s_e_i[1]), s_e_i[2]
        for father, son in replacement.items():
            if ph in son:
                ph = father
                break
        labels.append((ph, sample_to_frame(start, is_start=True), sample_to_frame(end, is_start=False)))

    return labels


def frame_to_sample(frame_no, rate=16000, hop=10, window=25):
    if frame_no == 0:
        return 0
    multiplier = rate // 1000
    return multiplier * window + (frame_no - 1) * hop * multiplier


def sample_to_frame(num, is_start, rate=16000, window=25, hop=10):
    multi = rate // (1000)
    if num < window * multi:
        return 0
    else:
        base_frame = (num - multi * window) // (multi * hop) + 1
        base_sample = frame_to_sample(base_frame)
        if is_start:
            if num - base_sample <= multi * hop // 2:
                return base_frame
            else:
                return base_frame + 1
        else:
            if num - base_sample <= multi * hop // 2:
                return base_frame - 1
            else:
                return base_frame


if __name__ == '__main__':
    # a = dl_model('train')
    # a.train()
    # a = dl_model('test')
    # a.test()
    a = dl_model('test_one')
    a.test_folder('trial')
