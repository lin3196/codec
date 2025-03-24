# Copyright (c) 2024 jishengpeng.
# Adapted under the MIT license.
# Source: https://github.com/jishengpeng/WavTokenizer

import os
import torch
from torch import nn
import math

from encoder.seanet import SEANetEncoder as Encoder
from quantizer.vq import ResidualVectorQuantizer as Quantizer
from decoder.backbone import VocosBackbone as Decoder
from decoder.head import ISTFTHead as Head


class WavTokenizer(nn.Module):
    """
    The Vocos class represents a Fourier-based neural vocoder for audio synthesis.
    This class is primarily designed for inference, with support for loading from pretrained
    model checkpoints. It consists of three main components: a feature extractor,
    a backbone, and a head.
    """

    def __init__(
        self,
        encoder,
        quantizer,
        decoder,
        head,
        sample_rate=16000
    ):
        super().__init__()

        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.head = head
        
        self.hop_length = math.prod(encoder.ratios)
        self.sample_rate = sample_rate
        
    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data


    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        
        z = self.encoder(audio_data)
        
        rvq_outputs = self.quantizer(
            z, n_quantizers
        )

        x = self.decoder(rvq_outputs["z"])
        
        audio_output = self.head(x)
        
        if audio_output.shape[-1] < length:
            audio_output = torch.nn.functional.pad(audio_output, [0, length-audio_output.shape[-1]])
        else:
            audio_output = audio_output[..., :length]
            
        outputs = {"audio": audio_output}
        outputs.update(rvq_outputs)
        
        return outputs
    

if __name__ == "__main__":
    from omegaconf import OmegaConf
    
    config = OmegaConf.load('config.yaml')
    
    encoder = Encoder(**config['model_config']['encoder'])
    quantizer = Quantizer(**config['model_config']['quantizer'])
    decoder = Decoder(**config['model_config']['decoder'])
    head = Head(**config['model_config']['head'])
    
    wavtokenizer = WavTokenizer(encoder, quantizer, decoder, head)
    
    x = torch.randn(1, 1, 16000*2)
    outs = wavtokenizer(x)
    
    y = outs['audio']
    print(y.shape)    