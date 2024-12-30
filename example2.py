import torch
import matplotlib.pyplot as plt

def plot_attention(attention:torch.Tensor, title:str):
    image_prefix = "images/" + title.replace(" ", "_")
    plt.figure(figsize=(10, 8))
    plt.imshow(attention[0].data.numpy(), aspect='auto', cmap='hot')
    plt.colorbar()
    plt.xlabel('Time Step')
    plt.ylabel('Time Step')
    plt.title(f"{title} - Attention Score - Heatmap")
    plt.savefig(f"{image_prefix}_Attention_Score_Heatmap.png", dpi=200)
    print(f"Saved to {image_prefix}_Attention_Score_Heatmap.png")

    plt.figure(figsize=(10, 5))
    plt.plot(attention[0, 6].data.numpy().T)
    plt.plot(attention[0, 2048].data.numpy().T)
    plt.plot(attention[0, 4095].data.numpy().T)
    # plt.legend([f'Time_Step_{i}' for i in range(4, 8)])
    plt.legend(['Time_Step_6', 'Time_Step_2048', 'Time_Step_4095'])
    plt.grid(True)
    plt.xlabel('Time Step')
    plt.ylabel('Attention Score')
    plt.title(f"{title} - Attention Score - Line Plot")
    plt.savefig(f"{image_prefix}_Attention_Score_LinePlot.png", dpi=200)
    print(f"Saved to {image_prefix}_Attention_Score_LinePlot.png")

from simplepe import SinusoidalPositionalEncoding, RotaryPositionalEncoding

def attention_score(x:torch.Tensor):
    return (x @ x.transpose(-2, -1)) / (x.shape[-1] ** 0.5)

B, T, D = 32, 4096, 512
# B, T, D = 1, 2048, 32
# B, T, D = 1, 2048, 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sinpe = SinusoidalPositionalEncoding(D, 2 * T).to(device)
x = torch.zeros(B, T, D).to(device)
# x1 = torch.randn(B, T, D).to(device)
x = sinpe(x)
attention = attention_score(x)
plot_attention(attention.cpu(), f"Sinusoidal Positional Encoding dim={D}")

rope = RotaryPositionalEncoding(D, 2 * T).to(device)
x = torch.ones(B, T, D).to(device)
x = rope(x)
attention = attention_score(x)
plot_attention(attention.cpu(), f"Rotary Positional Encoding dim={D}")
