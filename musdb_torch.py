
import torch
from torch.utils.data import Dataset
import numpy as np
import random

import musdb

class MUSDBSpectrogram(Dataset):
    def __init__(self, type, audio_params):
        self.musdb = musdb.DB(download=True, subsets=type, root="data/musdb")
        self.n_fft = audio_params["n_fft"]
        self.hop_length = audio_params["hop_length"]
        self.F = audio_params["F"]
        self.T = audio_params["T"]
        self.window = torch.hann_window(audio_params['n_fft'])

        # Number of samples to get FxT output https://pytorch.org/docs/stable/generated/torch.stft.html
        self.L = (self.T-1)*self.hop_length
        print(f"Num samples in chunk: {self.L}")

    def __len__(self):
        return len(self.musdb)

    def __getitem__(self, idx: int):
        track = self.musdb[idx]

        track.chunk_duration = self.L / track.rate
        track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)

        mix_raw = torch.from_numpy(track.audio.T.astype(np.float32))
        vocals_raw = torch.from_numpy(track.targets['vocals'].audio.T.astype(np.float32))
        accompaniment_raw = torch.from_numpy(track.targets['accompaniment'].audio.T.astype(np.float32))

        # Create spectrogram using STFT
        mix = torch.stft(mix_raw, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)
        mix = mix[:, :self.F, :]
        mix = mix.abs()

        vocals = torch.stft(vocals_raw, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)
        vocals = vocals[:, :self.F, :]
        vocals = vocals.abs()

        accompaniment = torch.stft(accompaniment_raw, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)
        accompaniment = accompaniment[:, :self.F, :]
        accompaniment = accompaniment.abs()

        return mix, vocals, accompaniment
