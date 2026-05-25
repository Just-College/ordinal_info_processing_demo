import numpy as np
import torch
from torch import nn


class CustomContinuousRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, tau=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tau = tau
        w_rec_std = 0.2 / np.sqrt(hidden_dim)
        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.W_rec = nn.Parameter(
            torch.normal(
                torch.zeros(hidden_dim, hidden_dim),
                w_rec_std * torch.ones(hidden_dim, hidden_dim),
            )
        )
        self.I_b = nn.Parameter(torch.zeros(hidden_dim))
        self.W_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_seq):
        B, T, _ = input_seq.shape
        device = input_seq.device
        x = torch.zeros(B, self.hidden_dim, device=device)
        r = torch.tanh(x)
        outputs = []
        for t in range(T):
            I_t = input_seq[:, t, :]
            dx = (
                -x + torch.matmul(r, self.W_rec.T) + self.W_in(I_t) + self.I_b
            ) / self.tau
            x = x + dx
            r = torch.tanh(x)
            y_t = self.W_out(r)
            outputs.append(y_t.unsqueeze(1))
        return torch.cat(outputs, dim=1)
