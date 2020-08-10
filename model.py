
import torch
import torch.nn as nn
import torch.nn.functional as F

channel_cnt = 256
k_s = 64
ctofc = 41
class cnn(nn.Module):

    def __init__(self):
        super(cnn, self).__init__()
        
        self.cv1 = nn.Conv1d(in_channels = 1, out_channels = channel_cnt, kernel_size = k_s)
        self.mp1 = nn.MaxPool1d(4)
        self.cv2 = nn.Conv1d(in_channels = channel_cnt, out_channels = channel_cnt, kernel_size = k_s)
        self.mp2 = nn.MaxPool1d(2)
        self.cv3 = nn.Conv1d(in_channels = channel_cnt, out_channels = channel_cnt, kernel_size = k_s)
        self.mp3 = nn.MaxPool1d(2)
        self.cv4 = nn.Conv1d(in_channels = channel_cnt, out_channels = channel_cnt, kernel_size = k_s)
        self.mp4 = nn.MaxPool1d(4)
        self.cv4 = nn.Conv1d(in_channels = channel_cnt, out_channels = channel_cnt, kernel_size = k_s)
        self.mp4 = nn.MaxPool1d(2)
        self.cv5 = nn.Conv1d(in_channels = channel_cnt, out_channels = channel_cnt, kernel_size = k_s)
        self.mp5 = nn.MaxPool1d(2)
        self.cv6 = nn.Conv1d(in_channels = channel_cnt, out_channels = 1, kernel_size = 2)
        self.mp6 = nn.MaxPool1d(2)

        self.dp1 = nn.Dropout(p = 0.2)

        self.bn2 = nn.BatchNorm1d(ctofc)
        self.fc2 = nn.Linear(ctofc, 50)
        self.dp2 = nn.Dropout(p = 0.2)
        #self.fc2.weight
        self.bn3 = nn.BatchNorm1d(50)
        self.fc3 = nn.Linear(50, 40)
        self.dp3 = nn.Dropout(p = 0.2)

        self.bn4 = nn.BatchNorm1d(40)
        self.fc4 = nn.Linear(40,30)


        #self.cv1 = nn.Conv1d(2000, 1, kernel_size = 200)
        
        #self.cv2 = nn.Conv1d()

    def forward(self, x):
        x.unsqueeze_(dim = 1)
        #x = self.bn1(x)
        x = self.cv1(x)
        x = torch.sin(x)
        x = self.mp1(x)

        x = self.cv2(x)
        x = torch.sin(x)
        #x = self.mp2(x)

        x = self.cv3(x)
        x = torch.sin(x)
        #x = self.mp3(x)

        x = self.cv4(x)
        x = torch.sin(x)
        x = self.mp4(x)

        x = self.cv5(x)
        x = torch.sin(x)
        #x = self.mp5(x)

        x = self.cv6(x)
        x = torch.sin(x)
        x = self.mp6(x)

        #y = x.sum(dim = 2)
        x.squeeze_(dim = 1)
        x = self.dp1(x)
        #print(x.shape)
        x = torch.sin(x)
        
        #print(x.shape)
        x = self.bn2(x)
        x = self.fc2(x)
        x = torch.sin(x)
        x = self.dp2(x)

        x = self.bn3(x)
        x = self.fc3(x)
        x = torch.sin(x)
        x = self.dp3(x)
        

        x = self.bn4(x)
        x = self.fc4(x)
        


        x = F.log_softmax(x, dim = 1)

        return x
        