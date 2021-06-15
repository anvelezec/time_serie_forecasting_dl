import torch
from torch.utils.data import Dataset
import numpy as np

def generate_time_serie(batch_size, n_steps):
    freq1, freq2, offset1, offset2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)

    series = 0.5 * np.sin((time - offset1) * (freq1 * 10 + 10))
    series += 0.5 * np.sin((time - offset2) * (freq2 * 20 + 20))
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5) # noise
    return series[..., np.newaxis].astype(np.float32)


class TSData(Dataset):
    def __init__(self, batch_size, n_steps) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.n_step = n_steps
        self.ts = generate_time_serie(batch_size, n_steps)

    def __len__(self) -> None:
        return len(self.ts)

    def __getitem__(self, index) -> None:
        x = self.ts[index, :self.n_step-1]
        y = self.ts[index, -1]
        return x, y