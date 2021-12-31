import librosa
import numpy as np
import pandas as pd
import glob
import torch
from torch.utils.data import Dataset

class AcousticEmissionDataset(Dataset):
    ' AE dataset. Child class of pytorch Dataset object.'
    
    # Constructor
    def __init__(self,dt,sr,low,high,num_bins,fft_units,
                 sig_len,fname,n_fft,hop_length):
        """
        
        dt (int): sample period = 1/sr
        sr (int): sample rate = 1/dt
        low(int): lower frequency threshold for filtering
        high(int): higher frequency threshold for filtering
        num_bins(int): number of bins for partial power calculation
        fft_units(int): convert Hz to smaller units, like kHz
        sig_len(int): number of samples in an AE signal
        fname(string): sample name to pull data from
        n_fft(int): number of samples per frame in STFT calculation
        hop_length(int): number of samples between frames in STFT calculation
        
        """
        self.dt = dt
        self.sr = sr
        self.low = low
        self.high = high
        self.num_bins = num_bins
        self.fft_units = fft_units
        self.sig_len = sig_len
        self.fname = fname
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Load in the raw acoustic event signals and the labels
        FNAME_raw = self.fname +'_waveforms'
        FNAME_filter = self.fname +'_filter'
        FNAME_label= self.fname +'_labels_channel_'
        raw = glob.glob("./"+fname+"_ae_data/"+FNAME_raw+".txt")[0]
        filter = glob.glob("./"+fname+"_ae_data/"+FNAME_filter+".csv")[0] 
        label = "./"+fname+"_ae_data/"+FNAME_label
        # Read in waveforms from each channel filtering out same events
        v0, ev = self._filter_ae(raw, filter, channel_num=0) # S9225
        v1, ev = self._filter_ae(raw, filter, channel_num=1) # S9225
        v2, ev = self._filter_ae(raw, filter, channel_num=2) # B1025
        v3, ev = self._filter_ae(raw, filter, channel_num=3) # B1025
        # Load in labeled data from sample folder
        # Labels for each channel's waveforms based on Spectral Clustering
        y0 = np.array(pd.read_csv(label+"A.csv").Cluster)
        y1 = np.array(pd.read_csv(label+"B.csv").Cluster)
        y2 = np.array(pd.read_csv(label+"C.csv").Cluster)
        y3 = np.array(pd.read_csv(label+"D.csv").Cluster)
        channels = [v0, v1, v2, v3]           # List of waveforms per channel
        channels_labels = [y0, y1, y2, y3]    # List of labels per channel
        x = [] 
        y = [] 
        for channel in channels: # append to single list in order v0,v1,v2,v3
            for waveform in channel:
                x.append(waveform)
        for channel in channels_labels:
            for label in channel:
                y.append(label)
            
        self.x = x                          # List of raw waveforms
        self.y = torch.tensor(y)            # tensor of waveform labels
        self.n_samples = self.y.shape[0]    # Number of samples/labels
        
    def __getitem__(self,index):
        """
        
        Function called when object is indexed. The transformation of data 
        occurs in this getter function. In other words, the constructor reads
        in the raw data filtered by hand, and this function contains the 
        sequence of remaining processing on the data to extract features used
        in ML models.
        
        index (int): the index of the sample and label
        
        return:
        (spectrogram,y): 2D stft array and label corresponding to single event
        
        """
        waveform = self.x[index]      
        
        # Compute spectrogram for the waveform
        spectrogram = self._compute_spectrogram(waveform)
        spectrogram = torch.FloatTensor(spectrogram)
        
        return spectrogram, self.y[index]
    
    def __len__(self):
        return self.n_samples
    
    def _read_ae_file2(self,ae_file,channel_num):
        """
        
        Read in text file with all ae hits and separate into waveforms based
        to the datasets signal length. 
        
        ae_file (string): file path to text file containing voltage vs time
        channel_num (int): channel to be read in, indexed from 0

        return:
        (sig, ev): where sig are signals and ev is event number indexed from 1
        
        """
        f = open(ae_file)
        lines = f.readlines()[1:]
        v1 = np.array([
            float(line.split()[channel_num]) for line in lines])
        f.close()
        sig = []
        for i in range(0,len(v1),self.sig_len):
            sig.append(v1[i:i+self.sig_len])
        ev = np.arange(len(sig))+1
        
        return sig,  ev


    def _filter_ae(self,ae_file,filter_csv,channel_num):
        """
        
        Reads in data and returns the events corresponding to the filtered meta
        data file.
        
        ae_file (string): file path to text file containing voltage vs time
        filter_csv (string): file path to meta data csv file of filtered events
        channel_num (int): channel to be read in, indexed from 0

        return: 
        (v1, event_num): where v1 is the within gauge signals and event_number 
        is the cooresponding event number indexed from 1
            
        """
        csv = pd.read_csv(filter_csv)
        ev = np.array(csv.Event)
        v1, _ = self._read_ae_file2(ae_file, channel_num)
        v1 = np.array(v1)
        v1 = v1[ev-1] # gets rid of some waveforms at this step
        
        return v1, ev
    
    def _compute_spectrogram(self,waveform):
        """        
        
        Computes 2D spectrogram from signal, using parameters set during init.
        Spectrogram is calculated using librosa short time fast fourier transf.
        Returned amplitude array is in logarithmic scale:
            
            = 20 * log(A/Ar) where Ar = 1.0
            
        STFT outputs only the positive side of the frequency, but it is not
        scaled according to Num of Samples / 2. The overlap is a hyperparameter
        that should be tuned; hop_length and n_fft / frame_size.
        
        waveform (1D arry floats): raw measurements, voltage vs. time
        
        returns: 
        (spectrogram): 2D stft np.ndarray [shape=(1 + n_fft/2, n_frames)
        
        """
        spectrogram = np.abs(librosa.stft(waveform,n_fft=self.n_fft,
                                         hop_length=self.n_fft+1))
        spectrogram = librosa.amplitude_to_db(spectrogram,ref=1.0)

        # Frequencies [0,sr/n_fft,2*sr/n_fft,....,sr/2]
        # shape -> 1d array of size = 1 + n_fft/2 (get rid of symmetric part)
        # https://librosa.org/doc/latest/generated/librosa.fft_frequencies.html
        freq = librosa.fft_frequencies(sr=self.sr,n_fft=self.n_fft)
        
        # Filter 
        if self.low is not None:
            spectrogram = spectrogram[np.where(freq > self.low)]
            freq = freq[np.where(freq > self.low)]

        if self.high is not None:
            spectrogram = spectrogram[np.where(freq < self.high)]
            freq = freq[np.where(freq < self.high)]

        return spectrogram
