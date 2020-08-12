import torch
from tqdm import tqdm
from glob import glob
from scipy.io import wavfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import librosa
import librosa.display

def specto_data_loader(files, data_tot):
    out = []
    idx = 0
    for file in tqdm(files):
        y, sr = librosa.load(file)
        mel = librosa.feature.melspectrogram(
            y = y,
            sr = sr,
            n_fft = 512,
            hop_length= 256
        )
        logamp = librosa.amplitude_to_db
        logspec = logamp(mel, ref = 1.0)
        # librosa.display.specshow(logspec, y_axis='mel', fmax=8000, x_axis='time')
        # plt.title('Mel Spectrogram')    
        # plt.colorbar(format='%+2.0f dB')
        # plt.show()
        # print(logspec.shape)
        # break
        out.append(logspec)
        idx += 1
        if idx > data_tot:
            break
        
    out = np.array(out)
    torch.save(torch.tensor(out), "spectro_preprocess.dat")
    
    return out

x_data = glob('data/train/*.wav')
data_tot = 2000000
#print(data_tot)
x_data = specto_data_loader(x_data, data_tot)