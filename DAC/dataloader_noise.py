"""
URGENT2 dataset: noise
"""
import random
import librosa
from torch.utils import data
import numpy as np
import soundfile as sf
from pathlib import Path
from omegaconf import OmegaConf
from collections import defaultdict


class URGENT2Dataset(data.Dataset):
    def __init__(self, cfg_yaml,
                 wav_len=4, num_per_epoch=10000, random_start=False,
                 fft_len=0.032,
                 selected_fs=[16000, 22050, 24000, 32000, 44100, 48000],
                 mode='train'):
        super().__init__()
        assert mode in ['train', 'validation']
        config = OmegaConf.load(cfg_yaml)
        self.wav_len = wav_len
        self.num_per_epoch = num_per_epoch
        self.random_start = random_start
        
        self.fft_len = fft_len
        self.selected_fs = selected_fs
        self.mode = mode
        
        self.speech_dic = {}
        for scp in config.speech_scps:
            with open(scp, "r") as f:
                for line in f:
                    uid, fs, audio_path = line.strip().split()
                    assert uid not in self.speech_dic, (uid, fs)
                    self.speech_dic[uid] = audio_path

        self.noise_dic = {}
        for scp in config.noise_scps:
            with open(scp, "r") as f:
                for line in f:
                    uid, fs, audio_path = line.strip().split()
                    assert uid not in self.noise_dic, (uid, fs)
                    self.noise_dic[uid] = audio_path
        self.noise_dic = dict(self.noise_dic)
        
        self.wind_noise_dic = {}
        for scp in config.wind_noise_scps:
            with open(scp, "r") as f:
                for line in f:
                    uid, fs, audio_path = line.strip().split()
                    assert uid not in self.wind_noise_dic, (uid, fs)
                    self.wind_noise_dic[uid] = audio_path
        self.wind_noise_dic = dict(self.wind_noise_dic)
        self.noise_dic.update(self.wind_noise_dic)
        
        self.rir_dic = None
        if config.rir_scps is not None:
            self.rir_dic = {}
            for scp in config.rir_scps:
                with open(scp, "r") as f:
                    for line in f:
                        uid, fs, audio_path = line.strip().split()
                        assert uid not in self.rir_dic, (uid, fs)
                        self.rir_dic[uid] = audio_path
        self.rir_dic = dict(self.rir_dic)
        
        self.meta = []
        if mode == 'train':
            meta_tsv = 'meta_new.tsv'
        elif mode == 'validation':
            meta_tsv = 'meta_filtered.tsv'
        else:
            raise ValueError
        
        print(f"You are using {meta_tsv}!")
        
        with open(Path(config.log_dir) / meta_tsv, "r") as f:
            headers = next(f).strip().split("\t")
            for line in f:
                info = dict(zip(headers, line.strip().split("\t")))
                if int(info['fs']) in self.selected_fs:
                    self.meta.append(info)

        if self.mode == 'train':
            self.meta2group()
        self.sample_data_per_epoch(mode)
    
    def sample_data_per_epoch(self, mode='train'):
        if mode == 'train':
            self.meta_selected = random.sample(self.meta[str(random.choice(self.selected_fs))], self.num_per_epoch)
        else:  # select fixed data when in validation or test
            self.meta_selected = self.meta[:self.num_per_epoch]
    
    def meta2group(self):
        grouped_info = defaultdict(list)
        for info in self.meta:
            fs = info["fs"]
            grouped_info[fs].append(info)
        self.meta = dict(grouped_info)
    
    def read_audio(self, filename, force_1ch=False, fs=None, start=0, stop=None):
        audio, fs_ = sf.read(filename, start=start, stop=stop, always_2d=True)
        audio = audio[:, :1].T if force_1ch else audio.T
        if fs is not None and fs != fs_:
            audio = librosa.resample(audio, orig_sr=fs_, target_sr=fs, res_type="soxr_hq")
            return audio, fs
        return audio, fs_

    def save_audio(self, audio, filename, fs):
        if audio.ndim != 1:
            audio = audio[0] if audio.shape[0] == 1 else audio.T
        sf.write(filename, audio, samplerate=fs)
     
     
    def __getitem__(self, idx):
        info = self.meta_selected[idx]
        
        uid = info["id"]
        # fs = int(info["fs"])
        fs = 16000
        snr = float(info["snr_dB"])  # SNR: [-5,20]

        rng = np.random.default_rng(int(uid.split("_")[-1]))

        noise = self.noise_dic[info["noise_uid"]]
        
        noise_info = sf.info(noise)
        noise_fs = noise_info.samplerate
        noise_length = int(noise_info.duration * noise_fs)
        
        if noise_length > noise_fs * 10:
            start = rng.integers(0, noise_length-noise_fs*10)
            stop = start + 10*noise_fs
            noise_sample = self.read_audio(noise, force_1ch=True, fs=fs, start=start, stop=stop)[0]
        else:
            noise_sample = self.read_audio(noise, force_1ch=True, fs=fs)[0]
                
        orig_len = noise_sample.shape[1]
    
        # select a segmen with a fixed duration in seconds
        if self.wav_len != 0:  # wav_len=0 means no cut or padding, use in test
            seg_len = self.wav_len*fs
            if seg_len < orig_len:
                start_point = rng.integers(0, orig_len-seg_len) if self.random_start else 0
                noise_sample = noise_sample[:, start_point: start_point + seg_len]
                
            elif seg_len > orig_len:
                pad_points = seg_len - orig_len
                noise_sample = np.pad(noise_sample, ((0, 0), (0, pad_points)), constant_values=0)

        # transform data type
        noise_sample = noise_sample.astype(np.float32)
        
        # normalization
        scale = 0.9 / (np.max(np.abs(noise_sample))+ 1e-12)

        noise_sample = noise_sample * scale

        info = {'id': uid, 'fs': fs, 'length': orig_len}

        return noise_sample, info
    
    
    def __len__(self):
        return len(self.meta_selected)

   
 
if __name__ == "__main__":
    import soundfile as sf
    from omegaconf import OmegaConf

    config = OmegaConf.load("config.yaml")

    train_dataset = URGENT2Dataset(**config['train_dataset'])
    train_dataloader = data.DataLoader(train_dataset, **config['train_dataloader'])
    
    valid_dataset = URGENT2Dataset(**config['validation_dataset'])
    valid_dataloader = data.DataLoader(valid_dataset, **config['validation_dataloader'])
    
    print(len(train_dataloader))
    print(len(valid_dataloader))
    
    for (speech, info) in train_dataloader:
        print(speech.shape)
        break
