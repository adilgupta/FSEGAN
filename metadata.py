"""
Converts raw TIMIT data into a pickle dump which can be used during training

NOTE: If n context frames are used, the size of the database will be n times larger than the base size.
Because each training example of (n+1) frames is generated separately and hence the same frame will be found and stored in (n+1) training examples
This works for smaller no. of context frames as the pickle dump is small enough to load into memory
THe advantage is that we get a slightly reduced train/test time since the dataloader does not need to construct
the training/testing every time when the model asks for a batch
For larger context, it would be better to shift the metadata code, which constructs the training examples from adjacent frames, to the dataloader
"""

import numpy as np
import pickle
import os
import scipy.io.wavfile as wav
from python_speech_features import logfbank, fbank
import json

#Ignore DS_Store files found on Mac
def listdir(pth):
	return [x for x in os.listdir(pth) if x != '.DS_Store']

#Convert from sample number to frame number
#e.g. sample 34*160 is in frame 1 assuming 25ms windows, 10 ms hop (assuming 0-indexing)
def sample_to_frame(num, rate=16000, window=25, hop=10):
	multi = rate//(1000)
	if num<window*multi:
		return 0
	else:
		return (num-multi*window)//(multi*hop)+1

class timit_data():

	def __init__(self, type_, config_file):

		self.config = config_file
		self.mode = type_
		self.db_path = config_file['dir']['dataset']

		#fold phones in list to the phone which is the key e.g. 'ao' is 'collapsed' into 'aa'
		self.replacement = {'aa':['ao'], 'ah':['ax','ax-h'],'er':['axr'],'hh':['hv'],'ih':['ix'],
		'l':['el'],'m':['em'],'n':['en','nx'],'ng':['eng'],'sh':['zh'],'pau':['pcl','tcl','kcl','bcl','dcl','gcl','h#','epi','q'],
		'uw':['ux']}

		self.pkl_name = self.db_path+self.mode+'_lstm'+'.pkl'

	#Generate and store pickle dump
	def gen_pickle(self):

		if os.path.exists(self.pkl_name):
			print("Found pickle dump for", self.mode)
			with open(self.pkl_name, 'rb') as f:
				return pickle.load(f)

		print("Generating pickle dump for", self.mode)

		to_return = [] #dictionary with key=phone and value=list of feature vectors with key phone in the centre frame
		base_pth = self.db_path+self.mode
		all_phones = set()
		num_distribution = {}

		for dialect in sorted(listdir(base_pth)):

			print("Dialect:", dialect)

			for speaker_id in sorted(listdir(os.path.join(base_pth, dialect))):

				data = sorted(os.listdir(os.path.join(base_pth, dialect, speaker_id)))
				wav_files = [x for x in data if x[-4:] == ".wav"] #all the .wav files

				for wav_file in wav_files:

					if wav_file in ['SA1.wav', 'SA2.wav']:
						continue

					wav_path = os.path.join(base_pth, dialect, speaker_id, wav_file)
					(rate,sig) = wav.read(wav_path)
					# sig ranges from -32768 to +32768 AND NOT -1 to +1
					feat, energy = fbank(sig, samplerate=rate, nfilt=self.config['feat_dim'], winfunc=np.hamming)
					feat_log_full = np.log(feat) #calculate log mel filterbank energies for complete file
					phenome_path = wav_path[:-7]+'PHN' #file which contains the phenome location data
					#phones in current wav file
					cur_phones = []

					with open(phenome_path, 'r') as f:
						a = f.readlines()

					for phenome in a:
						s_e_i = phenome[:-1].split(' ') #start, end, phenome_name e.g. 0 5432 'aa'
						start, end, ph = int(s_e_i[0]), int(s_e_i[1]), s_e_i[2]

						#collapse into father phone
						for father, list_of_sons in self.replacement.items():
							if ph in list_of_sons:
								ph = father
								break
						#update distribution
						all_phones.add(ph)
						if ph not in num_distribution.keys():
							num_distribution[ph] = 0
						num_distribution[ph] += 1

						#take only the required slice from the complete filterbank features
						feat_log = feat_log_full[sample_to_frame(start):sample_to_frame(end)+1]

						for i in range(feat_log.shape[0]):
							cur_phones.append((ph, feat_log[i]))

					to_return.append(cur_phones)
		#Dump pickle
		with open(self.pkl_name, 'wb') as f:
			pickle.dump(to_return, f)
			print("Dumped pickle")

		if self.mode == 'TRAIN':

			num_distribution = {k:1/v for k,v in num_distribution.items()}
			total_ph = sum(num_distribution.values())
			num_distribution = {k:v/total_ph for k,v in num_distribution.items()}
			#Dump mapping from id to phone. Used to convert NN output back to the phone it predicted
			phones_to_id = {}
			for ph in sorted(all_phones):
				phones_to_id[ph] = (len(phones_to_id), num_distribution[ph])

			phones_to_id['PAD'] = (len(phones_to_id), 0)
			#Dump this mapping
			fname = self.config['dir']['dataset']+'lstm_mapping.json'
			with open(fname, 'w') as f:
				json.dump(phones_to_id, f)

		return to_return


if __name__ == '__main__':

	config_file = {'dir':{'dataset':'./timit/data/'}, 'feat_dim':26}
	a = timit_data('TRAIN', config_file)
	a.gen_pickle()
