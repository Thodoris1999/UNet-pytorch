
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

# Default audio parameters to use.
AUDIO_PARAMS: Dict = {
    "instrument_list": ("vocals", "accompaniment"),
    "sample_rate": 44100,
    "n_fft": 4096,
    "hop_length": 1024,
    "T": 256, # time bins of STFT
    "F": 1024, # frequency bins of STFT
}

device = "cuda" if not torch.cuda.is_available() else "cpu"

def extend_mask(spec, params:Dict = AUDIO_PARAMS):
    n_extra_row = params["n_fft"] // 2 + 1 - params["F"]
    spec_shape = spec.size()
    extension_row = torch.zeros((spec_shape[0], spec_shape[1], 1, spec_shape[-1]), device=device)
    extension = torch.tile(extension_row, [1, 1, n_extra_row, 1])
    spec_pred = torch.cat([spec, extension], dim=2)
    return spec_pred


def main(args):
    params = AUDIO_PARAMS
    n_fft = params["n_fft"]
    F = params["F"]
    T = params["T"]
    hop_length = params['hop_length']
    window = torch.hann_window(params['n_fft'])
    # More iterations for higher quality ISTFT
    MAX_ITER_ISTFT = 50

    # read file
    waveform, sample_rate = torchaudio.load(args.input)
    if sample_rate != AUDIO_PARAMS["sample_rate"]:
        print(f"Only {AUDIO_PARAMS['sample_rate']}Hz sample rate is supported")
        exit(-1)

    # load accompaniment model
    accomp_checkpoint = torch.load(args.accomp_model_name)
    accomp_model = UNet()
    accomp_model.load_state_dict(accomp_checkpoint['model_state_dict'])
    accomp_model.to(device)
    accomp_model.eval()
    
    # load vocal model
    voc_checkpoint = torch.load(args.voc_model_name)
    voc_model = UNet()
    voc_model.load_state_dict(voc_checkpoint['model_state_dict'])
    voc_model.to(device)
    voc_model.eval()

    output_dir = os.path.splitext(args.input)[0]
    accomp_file = os.path.join(output_dir, "accomp.wav")
    voc_file = os.path.join(output_dir, "voc.wav")
    if (not os.path.exists(output_dir)):
        os.makedirs(output_dir)

    with torch.no_grad():
        # Create spectrogram using STFT
        mix = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        mix = mix[:, :F, :]
        mix = mix.abs()
        mix = torch.unsqueeze(mix, 0)
        print(mix.size())
        # pad to multiple of T
        mix_shape = mix.size()
        Tall = mix_shape[3]
        Tpadded = int(np.ceil(Tall / float(T))) * T
        n_extra_col = Tpadded - Tall
        extension_column = torch.zeros((mix_shape[0], mix_shape[1], mix_shape[2], 1), device=device)
        extension = torch.tile(extension_column, [1, 1, 1, n_extra_col])
        mix_padded = torch.cat([mix, extension], dim=3)
        print(mix_padded.size())

        wav_frame_size = (T-1) * hop_length
        frame_count = int(Tpadded / T)
        wav_size = wav_frame_size * frame_count
        vocal_wav = torch.zeros([2, wav_size])
        print(vocal_wav.size())
        accomp_wav = torch.zeros([2, wav_size])
        for i in range(frame_count):
            frame_start = i*T
            frame_end = (i+1) * T
            mix_frame = mix_padded[:, :, :, frame_start:frame_end]

            voc_pred = voc_model(mix_frame)
            accomp_pred = accomp_model(mix_frame)
            voc_pred = extend_mask(voc_pred)
            accomp_pred = extend_mask(accomp_pred)
            voc_pred = torch.squeeze(voc_pred)
            accomp_pred = torch.squeeze(accomp_pred)

            frame_start_wav = i*wav_frame_size
            frame_end_wav = (i+1) * wav_frame_size
            vocal_frame_wav = griffin_lim(voc_pred, max_iter=MAX_ITER_ISTFT, n_fft=AUDIO_PARAMS['n_fft'], hop_length=AUDIO_PARAMS['hop_length'], window=window)
            vocal_wav[:, frame_start_wav:frame_end_wav] = vocal_frame_wav
            accomp_frame_wav = griffin_lim(accomp_pred, max_iter=MAX_ITER_ISTFT, n_fft=AUDIO_PARAMS['n_fft'], hop_length=AUDIO_PARAMS['hop_length'], window=window)
            accomp_wav[:, frame_start_wav:frame_end_wav] = accomp_frame_wav

        torchaudio.save(voc_file, vocal_wav, AUDIO_PARAMS['sample_rate'])
        torchaudio.save(accomp_file, accomp_wav, AUDIO_PARAMS['sample_rate'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--accomp_model_name", "-a", default="accomp_unet.pt")
    parser.add_argument("--voc_model_name", "-v", default="voc_unet.pt")
    parser.add_argument("--input", "-i", default="accomp_unet.pt", help="Input audio file")
    parser.add_argument("--batch_size", default=4, type=int, help="Number of test images to generate reconstructions from")
    args = parser.parse_args()
    main(args)
