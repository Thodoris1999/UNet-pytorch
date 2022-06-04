# UNet-pytorch
Pytorch implementation of U-Net for music source separation.

## Installing Dependencies
- FFmpeg for audio I/O
`sudo apt install ffmpeg`
- python module requirements
```
python3 -m venv env
. env/bin/activate
pip install -r requirements.txt
```
### Training
`python train.py -v voc_weights.pt -a accompaniment_weights.py`
### Testing
`python separate_musdb.py -v voc_weights.pt -a accompaniment_weights.py`
### Separating songs
`python song_separator.py -v voc_weights.pt -a accompaniment_weights.py -i song.mp3`

Use the `-h` flag for more information on the command line arguments
