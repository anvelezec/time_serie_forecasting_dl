# Time serie forecasting with deep learning

Exploration of different deep learning techniques to deal with time series forecasting. Covering RNN such as basic RRN, GRU, LSTS, attention layers and some of latest papers about this topic.


## Data generation
To train the NN`s we use a data generation with sin waves and random noise. This generator is absorved by a pytorch Data Generator. Example usage:

```python
from data import TSData

training_data = TSData(batch_size=7000, n_steps=24)
```

## Models
