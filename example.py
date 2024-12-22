import torch
from einops import rearrange # pip install einops
import matplotlib.pyplot as plt # pip install matplotlib
from pathlib import Path

Path("images").mkdir(exist_ok=True)

def plot_encoding(y:torch.Tensor, title:str):
    image_prefix = "images/" + title.replace(" ", "_")
    plt.figure(figsize=(15, 5))
    plt.plot(y[0, :, 4:8].data.numpy())
    plt.legend([f'Dimension_{i}' for i in range(4, 8)])
    plt.grid(True)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(f"{title} - Line Plot")
    plt.savefig(image_prefix + "_LinePlot.png", dpi=200)
    print(f"Saved to {image_prefix}_LinePlot.png")

    # plot the tensor
    plt.figure(figsize=(15, 5))
    plt.imshow(y[0].T.data.numpy(), aspect='auto')
    plt.colorbar()
    plt.xlabel('Time Step')
    plt.ylabel('Dimension')
    plt.title(f"{title} - Heatmap")
    plt.savefig(image_prefix + "_Heatmap.png", dpi=200)
    print(f"Saved to {image_prefix}_Heatmap.png")

from simplepe import SinusoidalPositionalEncoding, RotaryPositionalEncoding

pe = SinusoidalPositionalEncoding(20, 100)
x = torch.zeros(1, 100, 20)
y = pe(x)
plot_encoding(y, "Sinusoidal Positional Encoding")

pe = RotaryPositionalEncoding(20, 100)
x = torch.ones(1, 100, 20)
y = pe(x)
plot_encoding(y, "Rotary Positional Encoding")

# multi head input
pe = SinusoidalPositionalEncoding(32, 200)
x = torch.zeros(1, 200, 8 * 32)
x = rearrange(x, 'B T (nH Hs) -> B nH T Hs', nH=8)
y = pe(x)
y = rearrange(y, 'B nH T Hs -> B T (nH Hs)')
plot_encoding(y, "Multi-Head Sinusoidal Positional Encoding")

pe = RotaryPositionalEncoding(32, 200)
x = torch.ones(1, 200, 8 * 32)
x = rearrange(x, 'B T (nH Hs) -> B nH T Hs', nH=8)
y = pe(x)
y = rearrange(y, 'B nH T Hs -> B T (nH Hs)')
plot_encoding(y, "Multi-Head Rotary Positional Encoding")
