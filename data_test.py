from data import TSData


def test_ts_data():
    batch_size = 7000
    n_steps = 24
    training_data = TSData(batch_size, n_steps)

    # Display image and label.
    train_features, train_labels = next(iter(training_data))
    assert len(train_features) + len(train_labels) == n_steps
    assert len(train_labels) == batch_size
