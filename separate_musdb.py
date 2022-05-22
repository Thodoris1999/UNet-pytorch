
import torch
from torch.utils.data import DataLoader
import torchaudio
from torch_specinv import griffin_lim
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from typing import Dict

from unet import UNet
from musdb_torch import MUSDBSpectrogram

# Default audio parameters to use.
AUDIO_PARAMS: Dict = {
    "instrument_list": ("vocals", "accompaniment"),
    "sample_rate": 44100,
    "n_fft": 4096,
    "hop_length": 1024,
    "T": 256, # time bins of STFT
    "F": 1024, # frequency bins of STFT
}

device = "cuda" if torch.cuda.is_available() else "cpu"

def extend_mask(spec, params:Dict = AUDIO_PARAMS):
    n_extra_row = params["n_fft"] // 2 + 1 - params["F"]
    spec_shape = spec.size()
    extension_row = torch.zeros((spec_shape[0], spec_shape[1], 1, spec_shape[-1]), device=device)
    extension = torch.tile(extension_row, [1, 1, n_extra_row, 1])
    spec_pred = torch.cat([spec, extension], dim=2)
    return spec_pred


def main(args):
    accomp_checkpoint = torch.load(args.accomp_model_name)
    accomp_model = UNet()
    accomp_model.load_state_dict(accomp_checkpoint['model_state_dict'])
    accomp_model.to(device)
    accomp_model.eval()
    
    voc_checkpoint = torch.load(args.voc_model_name)
    voc_model = UNet()
    voc_model.load_state_dict(voc_checkpoint['model_state_dict'])
    voc_model.to(device)
    voc_model.eval()

    test_data = MUSDBSpectrogram(type='test', audio_params=AUDIO_PARAMS, include_name=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    
    if (not os.path.exists(args.output_dir)):
        os.makedirs(args.output_dir)

    i = 0
    with torch.no_grad():
        for mix,vocals,accompaniment,name in test_dataloader:
            mix, vocals, accompaniment = mix.to(device), vocals.to(device), accompaniment.to(device)
            voc_pred = voc_model(mix)
            accomp_pred = accomp_model(mix)
            voc_pred = extend_mask(voc_pred)
            accomp_pred = extend_mask(accomp_pred)
            for (voc,accomp,track_name) in zip(voc_pred, accomp_pred, name):
                mus_dir = os.path.join(args.output_dir, track_name)
                if (not os.path.exists(mus_dir)):
                    os.makedirs(mus_dir)
                accomp_file = os.path.join(mus_dir, "accomp.wav")
                voc_file = os.path.join(mus_dir, "voc.wav")
                i += 1

                window = torch.hann_window(AUDIO_PARAMS['n_fft']).to(device)
                vocal_wav = griffin_lim(voc, n_fft=AUDIO_PARAMS['n_fft'], hop_length=AUDIO_PARAMS['hop_length'], window=window)
                torchaudio.save(voc_file, vocal_wav.cpu(), AUDIO_PARAMS['sample_rate'])
                accomp_wav = griffin_lim(accomp, n_fft=AUDIO_PARAMS['n_fft'], hop_length=AUDIO_PARAMS['hop_length'], window=window)
                torchaudio.save(accomp_file, accomp_wav.cpu(), AUDIO_PARAMS['sample_rate'])




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--accomp_model_name", "-a", default="accomp_unet.pt")
    parser.add_argument("--voc_model_name", "-v", default="voc_unet.pt")
    parser.add_argument("--output_dir", "-o", default="pred")
    parser.add_argument("--batch_size", default=4, type=int, help="Number of test images to generate reconstructions from")
    args = parser.parse_args()
    main(args)
