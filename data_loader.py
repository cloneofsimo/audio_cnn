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
import random
import torchvision.transforms as transforms

def data_loader(files, data_tot):
    out = []
    idx = 0
    for file in tqdm(files):
        #print(file)
        fs, data = wavfile.read(file)
        out.append(data)
        idx += 1
        
        if idx > data_tot:
            break  
    out = np.array(out)
    return out

def specto_data_loader(files, data_tot):
    out = []
    idx = 0
    for file in tqdm(files):
        y, sr = librosa.load(file)
        mel = librosa.feature.melspectrogram(
            y = y,
            n_fft = 1024,
            n_mels = 256,
            hop_length= 512
        )
        logamp = librosa.amplitude_to_db
        logspec = logamp(mel, ref = 1.0)
        librosa.display.specshow(logspec, y_axis='mel', fmax=8000, x_axis='time')
        # plt.title('Mel Spectrogram')    
        # plt.colorbar(format='%+2.0f dB')
        # plt.show()
        # break
        out.append(logspec)
        idx += 1
        if idx > data_tot:
            break
    out = np.array(out)
    return out

device = torch.device("cuda:0")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, rng):
        x_data = glob('data/train/*.wav')
        data_tot = rng[1]
        #print(data_tot)
        #x_data = specto_data_loader(x_data, data_tot)

        #if not spectro:

        # x_data = data_loader(x_data, data_tot)
        # x_data = x_data[:, ::8]/30000 # 매 8번째 데이터만 사용
        #  # 최대값 30,000 을 나누어 데이터 정규화
        # self.x_data = torch.tensor(x_data).float()

        #if spectro:

        x_data = torch.load("spectro_preprocess.dat")
        self.x_data = x_data[rng[0]:rng[1]]
        
        
        
        # 정답 값을 불러옵니다
        y_data = pd.read_csv('data/train_answer.csv', index_col=0)
        y_data = y_data.values
        y_data = y_data[rng[0] : rng[1]]
        
        #y_data = torch.tensor(4*y_data).long().to(device)
        #y_data = (y_data > 0).long().to(device)

        
        #self.x_data = x_data
        self.y_data = torch.tensor(y_data)
        self.len = rng[1] - rng[0]
    
    def __len__(self):
        return 2*self.len
    
    def __getitem__(self, idx):
        if idx >= self.len:
            p = random.random()
            if p > 0.3:
                q = 1 - p
                idx1 = random.randint(0, self.len)%self.len
                idx2 = random.randint(0, self.len)%self.len
                
                return p*self.x_data[idx1] + q*self.x_data[idx2], p*self.y_data[idx1] + q*self.y_data[idx2]
            else:
                idx -= self.len
        return self.x_data[idx], self.y_data[idx]

