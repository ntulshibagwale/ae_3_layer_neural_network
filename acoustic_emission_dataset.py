import librosa
import numpy as np
import pandas as pd
import glob
import torch
from ae_measure2 import *
from torch.utils.data import Dataset

class AcousticEmissionDataset(Dataset):
    
    def __init__(self,dt,sr,low,high,num_bins,fft_units,
                 sig_len,fname,frame_size):
        
        self.dt = dt
        self.sr = sr
        self.low = low
        self.high = high
        self.num_bins = num_bins
        self.fft_units = fft_units
        self.sig_len = sig_len
        self.fname = fname
        self.frame_size = frame_size
        
        
        # Load in the raw acoustic event signals and the labels
        FNAME_raw = self.fname +'_waveforms'
        FNAME_filter = self.fname +'_filter'
        FNAME_labels= self.fname +'_labels_channel_'
        
        raw = glob.glob("./data/"+FNAME_raw+".txt")[0]
        filter = glob.glob("./data/"+FNAME_filter+".csv")[0] 
        
        # Waveforms from each channel
        v0, ev = filter_ae(raw, filter, channel_num=0, 
                           sig_length=self.sig_len) # S9225
        v1, ev = filter_ae(raw, filter, channel_num=1,
                           sig_length=self.sig_len) # S9225
        v2, ev = filter_ae(raw, filter, channel_num=2,
                           sig_length=self.sig_len) # B1025
        v3, ev = filter_ae(raw, filter, channel_num=3,
                           sig_length=self.sig_len) # B1025
        
        # Labels for each channel's waveforms based on Spectral Clustering
        y0 = np.array(pd.read_csv("./data/"+FNAME_labels+"A.csv").Cluster)
        y1 = np.array(pd.read_csv("./data/"+FNAME_labels+"B.csv").Cluster)
        y2 = np.array(pd.read_csv("./data/"+FNAME_labels+"C.csv").Cluster)
        y3 = np.array(pd.read_csv("./data/"+FNAME_labels+"D.csv").Cluster)

        channels = [v0, v1, v2, v3]           # List of waveforms per channel
        channels_labels = [y0, y1, y2, y3]    # List of labels per channel
        
        x = [] 
        y = [] 
        for channel in channels:
            for waveform in channel:
                x.append(waveform)
        for channel in channels_labels:
            for label in channel:
                y.append(label)
            
        self.x = x                          # List of raw waveforms
        self.y = torch.tensor(y)            # List of waveform labels
        self.n_samples = self.y.shape[0]    # Number of samples/labels
        
    def __getitem__(self,index):
        
        # Compute spectrogram for the waveform
        waveform = self.x[index]
        spectrogram = self._compute_spectrogram(waveform)
        spectrogram = torch.FloatTensor(spectrogram)
        
        return spectrogram, self.y[index]
    
    def __len__(self):
        return self.n_samples
    
    def _compute_spectrogram(self,waveform):
        """        
        
        Computes 2D spectrogram from signal, using parameters set during init.
        Spectrogram is calculated using librosa short time fast fourier transf.
        Returned amplitude array is in logarithmic scale:
            
            = 20 * log(A/Ar) where Ar = 1.0
            
        STFT outputs only the positive side of the frequency, but it is not
        scaled according to Num of Samples / 2.
            
        
        """
    
        spectrogram = np.abs(librosa.stft(waveform,n_fft=self.frame_size,
                                         hop_length=self.frame_size+1))
            
        spectrogram = librosa.amplitude_to_db(spectrogram,ref=1.0)

        # Frequencies [0,sr/n_fft,2*sr/n_fft,....,sr/2]
        # shape -> 1d array of size = 1 + n_fft/2 (get rid of symmetric part)
        # https://librosa.org/doc/latest/generated/librosa.fft_frequencies.html
        freq = librosa.fft_frequencies(sr=self.sr,n_fft=self.frame_size)
        
        # Filter 
        if self.low is not None:
            spectrogram = spectrogram[np.where(freq > self.low)]
            freq = freq[np.where(freq > self.low)]

        if self.high is not None:
            spectrogram = spectrogram[np.where(freq < self.high)]
            freq = freq[np.where(freq < self.high)]

        return spectrogram
    
    
if __name__=='__main__':
    
    # Load in the raw acoustic event signals and the labels
    DT = 10**-7              # [seconds] ; sample period / time between samples
    SR = 1/DT                # [Hz] ; sampling rate
    LOW = 200*10**3          # [Hz] ; low frequency cutoff
    HIGH = 800*10**3         # [Hz] ; high frequency cutoff
    NUM_BINS = 26            # Number of bins for partial power feature vector
    FFT_UNITS = 1000         # FFT outputs in Hz, this converts to kHz
    SIG_LEN = 1024           # [samples / signal] ;
    FNAME = '210330-1'       # File name ; '210308-1','210316-1','210330-1'
    FRAME_SIZE = 256         # Number of samples per frame of stft

    ae_data=AcousticEmissionDataset(DT,SR,LOW,HIGH,NUM_BINS,FFT_UNITS,SIG_LEN,FNAME,FRAME_SIZE)
    ae_data[1]