import numpy as np
import pandas as pd
import glob
import os
import pdb

if __name__=="__main__":
    df = np.array(pd.read_csv('./timit/train_data.csv'))
    count = 0
    #pdb.set_trace()
    #cf = df[np.logical_and(np.logical_and('SX' in df['filename'], '.wav' in df['filename']), df['is_converted_audio']==True)]
    # 4 - file name
    # 7 - is_converted_audio
    # 8 - is audio
    # 10 - is_phonetic_file
    for i in range(df.shape[0]):
        if np.isnan(df[i,0]):
            continue
        elif ('SX' in df[i,4] and '.PHN' in df[i,4]):
            tmp = df[i,5].split('/')
            #os.system("scp /media/Sharedata/adil/timit/data/" + df[i, 5] + " /media/Sharedata/adil/timit/phone_seq/" + tmp[-3] + "_" + tmp[-2]+"_" + tmp[-1])
            count+=1
            print(count)
    pdb.set_trace()
