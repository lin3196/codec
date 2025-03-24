# Copyright (c) 2025 jishengpeng.
# Adapted under the MIT license.
# Source: https://github.com/jishengpeng/WavTokenizer

import torch
import torch.nn as nn


    
class ISTFTHead(nn.Module):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int):
        super().__init__()
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, D, L), where B is the batch size,
                        D denotes the model dimension, and L is the sequence length.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x.transpose(1, 2)).transpose(1,2)
        
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        # recalculating phase here does not produce anything new
        # only costs time
        # phase = torch.atan2(y, x)
        # S = mag * torch.exp(phase * 1j)
        # better directly produce the complex value 
        S = mag * (x + 1j * y)
        audio = torch.istft(S, self.n_fft, self.hop_length, self.n_fft, torch.hann_window(self.n_fft).to(x.device))
        
        return audio
    

if __name__ == "__main__":
    model = ISTFTHead(768, 1280, 320)
    
    x = torch.randn(1, 768, 126)
    y = model(x)
    
    print(y.shape)