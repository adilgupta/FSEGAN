import glob
import numpy as np
import pandas as pd
import pdb
import os

if __name__ == "__main__":
    file_df = pd.read_csv('./timit/train_data.csv')
    # going to use SX files, phonetically diverse set
    #pdb.set_trace()

    #file_df = file_df.iloc[np.logical_not(np.isnan(file_df['index']))]
    for i in range(file_df.shape[0]):
        #print(file_df['path_from_data_dir'][i], i)
        if(not np.isnan(file_df['index'][i])):
            if(not os.path.isfile('./timit/data/' + file_df['path_from_data_dir'][i])):
                print('not there')
                continue
        if(file_df['is_audio'][i]==True and file_df['is_converted_audio'][i]==False):
            if(not os.path.isfile('./timit/data/' + file_df['path_from_data_dir'][i][:-4]+'.WAV.wav')):
                print('error')
                pdb.set_trace()
    pdb.set_trace()
    pdb.set_trace()
