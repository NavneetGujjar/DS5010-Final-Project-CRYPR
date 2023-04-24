import numpy as np

def train_test_split(data, target, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
        
    # Shuffle the indices
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    # Calculate the split index
    split_idx = int(len(data) * (1 - test_size))

    # Split the data and target arrays using the shuffled indices
    train_data = data[indices[:split_idx]]
    test_data = data[indices[split_idx:]]
    train_target = target[indices[:split_idx]]
    test_target = target[indices[split_idx:]]
    
    return train_data, test_data, train_target, test_target
