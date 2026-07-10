"""Dataset classes for language model training."""

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    """Dataset for language modeling that returns (input, target) token sequences.
    
    This dataset reads from a memory-mapped file containing tokenized text and
    returns overlapping sequences for next-token prediction.
    
    Args:
        data_path: Path to the .bin file containing np.uint16 tokens
        block_size: Length of each sequence (context length)
    """
    
    def __init__(self, data_path, block_size):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        # Number of valid starting positions (need block_size + 1 tokens total)
        self.length = len(self.data) - block_size
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """
        Returns:
            x: Input token sequence of length block_size
            y: Target token sequence of length block_size (shifted by 1)
        """
        # Get input and target sequences
        x_data = self.data[idx:idx + self.block_size].astype(np.int64)
        x = torch.from_numpy(x_data)
        y_data = self.data[idx + 1:idx + 1 + self.block_size].astype(np.int64)
        y = torch.from_numpy(y_data)
        return x, y

