import argparse
from musdb_torch import MUSDBSpectrogram
from unet import UNet
from torch_specinv import griffin_lim
from typing import Dict
import numpy as np
import os
import time

from torch.utils.data import DataLoader
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# Default audio parameters to use.
AUDIO_PARAMS: Dict = {
    "instrument_list": ("vocals", "accompaniment"),
    "sample_rate": 44100,
    "n_fft": 4096,
    "hop_length": 1024,
    "T": 256, # time bins of STFT
    "F": 1024, # frequency bins of STFT
}

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight)

def train_batch(dataloader, voc_model, accomp_model, voc_optim, accomp_optim):
    size = len(dataloader.dataset)
    voc_model.train()
    accomp_model.train()
    for batch, (mix,vocals,accompaniment) in enumerate(dataloader):
        mix, vocals, accompaniment = mix.to(device), vocals.to(device), accompaniment.to(device)
        voc_pred = voc_model(mix)
        voc_loss = F.l1_loss(voc_pred, vocals)
        accomp_pred = accomp_model(mix)
        accomp_loss = F.l1_loss(accomp_pred, accompaniment)

        voc_optim.zero_grad()
        voc_loss.backward()
        voc_optim.step()
        accomp_optim.zero_grad()
        accomp_loss.backward()
        accomp_optim.step()

        if batch % 10 == 0:
            voc_loss, accomp_loss, current = voc_loss.item(), accomp_loss.item(), batch*len(mix)
            print(f"Vocals loss: {voc_loss:>7f}, accompaniment loss: {accomp_loss:>7f} [{current}/{size}]")

    return voc_loss, accomp_loss


def test_batch(dataloader, voc_model, accomp_model):
    num_batches = len(dataloader)
    voc_model.eval()
    accomp_model.eval()
    total_voc_loss = 0
    total_accomp_loss = 0

    with torch.no_grad():
        for mix,vocals,accompaniment in dataloader:
            mix, vocals, accompaniment = mix.to(device), vocals.to(device), accompaniment.to(device)
            voc_pred = voc_model(mix)
            voc_loss = F.l1_loss(voc_pred, vocals)
            accomp_pred = accomp_model(mix)
            accomp_loss = F.l1_loss(accomp_pred, accompaniment)

            total_voc_loss += voc_loss
            total_accomp_loss += accomp_loss

    total_voc_loss /= num_batches
    total_accomp_loss /= num_batches
    return total_voc_loss, total_accomp_loss


def train(voc_model, accomp_model, batch_size, lr, train_dataloader, test_dataloader, voc_checkpoint, accomp_checkpoint, epochs=16, retrain=False):
    print("Using {} device".format(device))
    voc_model.to(device)
    accomp_model.to(device)

    voc_params = None
    if os.path.exists(voc_checkpoint) and not retrain:
        voc_params = torch.load(voc_checkpoint)
        voc_model_state, voc_optimizer_state = voc_params['model_state_dict'], voc_params['optimizer_state_dict']
        voc_model.load_state_dict(voc_model_state)
        voc_optimizer = optim.Adam(voc_model.parameters(), lr=lr)
        voc_optimizer.load_state_dict(voc_optimizer_state)
    else:
        voc_optimizer = optim.Adam(voc_model.parameters(), lr=lr)
    accomp_params = None
    if os.path.exists(accomp_checkpoint) and not retrain:
        accomp_params = torch.load(accomp_checkpoint)
        accomp_model_state, accomp_optimizer_state = accomp_params['model_state_dict'], accomp_params['optimizer_state_dict']
        accomp_model.load_state_dict(accomp_model_state)
        accomp_optimizer = optim.Adam(accomp_model.parameters(), lr=lr)
        accomp_optimizer.load_state_dict(accomp_optimizer_state)
    else:
        accomp_optimizer = optim.Adam(accomp_model.parameters(), lr=lr)

    voc_min_loss = voc_params['test_loss'] if voc_params is not None else None
    accomp_min_loss = accomp_params['test_loss'] if accomp_params is not None else None
    total_losses = np.zeros((epochs,2))
    print("---------------Beginning training------------------")

    for t in range(epochs):
        print(f"Epoch {t}\n-----------------------")
        print(time.time())
        train_batch(train_dataloader, voc_model, accomp_model, voc_optimizer, accomp_optimizer)
        voc_loss, accomp_loss = test_batch(test_dataloader, voc_model, accomp_model)

        total_losses[t] = [voc_loss.item(), accomp_loss.item()]
        if voc_min_loss is None or voc_loss < voc_min_loss:
            voc_min_loss = voc_loss
            torch.save({
                'model_state_dict': voc_model.state_dict(),
                'optimizer_state_dict': voc_optimizer.state_dict(),
                'test_loss': voc_loss,
            }, voc_checkpoint)
            print(f'Saved best vocal model weights at epoch {t} with loss {voc_loss:>5f}')
        if accomp_min_loss is None or accomp_loss < accomp_min_loss:
            accomp_min_loss = accomp_loss
            torch.save({
                'model_state_dict': accomp_model.state_dict(),
                'optimizer_state_dict': accomp_optimizer.state_dict(),
                'test_loss': accomp_loss,
            }, accomp_checkpoint)
            print(f'Saved best accompaniment model weights at epoch {t} with loss {accomp_loss:>5f}')

    print("Done!")
    return total_losses

def main(args):
    voc_model = UNet()
    accomp_model = UNet()
    train_data = MUSDBSpectrogram(type='train', audio_params=AUDIO_PARAMS)
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    test_data = MUSDBSpectrogram(type='test', audio_params=AUDIO_PARAMS)
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)
    losses = train(voc_model, accomp_model, 4, 1e-4, train_dataloader, test_dataloader, args.voc_model_name,
                    args.accomp_model_name, epochs=256)
    print(losses)
"""
    model = UNet()
    model.train()
    model = model.to(device)
    model.apply(init_weights)

    train_data = MUSDBSpectrogram(audio_params=AUDIO_PARAMS)
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    for data in train_dataloader:
        mix, vocals, _ = data
        #mix, vocals = mix.to(device), vocals.to(device)
        #voc = vocals[0]
        # discard phase information
        #vocal_wav = griffin_lim(voc).cpu()

        vocals_pred = vocals
        vocals_shape = vocals.size()
        window = torch.hann_window(AUDIO_PARAMS['n_fft']).to(device)
        n_extra_row = AUDIO_PARAMS["n_fft"] // 2 + 1 - AUDIO_PARAMS["F"]
        extension_row = torch.zeros((vocals_shape[0], vocals_shape[1], 1, vocals_shape[-1]), device=device)
        extension = torch.tile(extension_row, [1, 1, n_extra_row, 1])
        vocals_pred = torch.cat([vocals_pred, extension], dim=2)
        vocal_wav = griffin_lim(vocals_pred[0], n_fft=AUDIO_PARAMS['n_fft'], hop_length=AUDIO_PARAMS['hop_length'], window=window).cpu()
        torchaudio.save('test.wav', vocal_wav, 44100)
        break
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc_model_name", "-v", default="voc_unet.pt")
    parser.add_argument("--accomp_model_name", "-a", default="accomp_unet.pt")
    args = parser.parse_args()
    main(args)
