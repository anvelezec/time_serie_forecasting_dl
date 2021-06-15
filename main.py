from torch.utils.data import DataLoader

from data import generate_time_serie
from data import TSData

n_steps = 50
ts = generate_time_serie(10000, n_steps + 1)


training_data = TSData(batch_size=7000, n_steps=24)
test_data = TSData(batch_size=1000, n_steps=24)


train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)