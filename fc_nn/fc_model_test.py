import torch
import numpy as np
from torch.nn.modules.linear import Linear

from fc_model import FullConnect

device = "cuda" if torch.cuda.is_available() else "cpu"

def test_fc_model():
    model = FullConnect(23, 10, 1).to(device)
    assert isinstance(model, torch.nn.Module)


def test_fc_model_basic_input():
    batch_size, input_size = np.random.randint(1,20, 2)
    x = torch.rand((batch_size, input_size, 1))
    model = FullConnect(input_size, 10, 1).to(device)
    y = model(x)
    assert batch_size == y.shape[0]
