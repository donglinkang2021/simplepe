# from simplepe.rope1 import RotaryPositionalEncoding
import torch
import simplepe

torch.manual_seed(1337)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
B, T, D = 32, 4096, 512
rope = simplepe.rope.RotaryPositionalEncoding(D, 2 * T).to(device)
rope1 = simplepe.rope1.RotaryPositionalEncoding(D, 2 * T).to(device)
x = torch.ones(B, T, D).to(device)
x1 = rope(x)
x2 = rope1(x)
print(torch.nn.functional.mse_loss(x1, x2).item())  # should be 0
