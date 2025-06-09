"""
V1 of the GPT data preparation based on the Book 'Build a Large Language Model' by Sebastian Raschka.
Implements a sliding window dataloader given a specific text file or text as a string.
"""

import tiktoken
from torch.utils.data import Dataset
import torch
from namespaces.test_input_types import TextInputType
from typing import Tuple


class GPTDataPrepV1(Dataset):
    """
    Child of the parent Dataset class.
    """

    def __init__(
        self,
        text_input_type: TextInputType,
        text_input: str,
        tokenizer,
        max_length: int,
        stride: int,
    ):
        """
        Class constructor.

        Args:
            text_input_type (TextInputType): The type of text input that is provided.
            text_input (str): The text input to be processed.
                If the TextInputType is FILE_PATH, then the text input is a path to a file.
                If the TextInputType is STRING, then the text input is a string.
            tokenizer (): The tokenizer to be used.
            max_length (int): The maximum length of the text input.
            stride (int): The stride of the sliding window.
                SLide basically controls the number of tokens by which the window jumps when it is moved. The granularity of
                the slide in the sliding window is the stride.
        """

        # Create a list of input and target ids.
        self.input_ids = []
        self.target_ids = []

        # Set the text input based in the input type we choose.
        if text_input_type == TextInputType.FILE_PATH:
            with open(text_input, "r") as f:
                text = f.read()
        elif text_input_type == TextInputType.STRING:
            text = text_input

        token_ids = tokenizer.encode(text)

        # Now we create the input and the target ids.
        # So we iterate from 0 to token_count - max_length, with a step of stride.
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.input_ids)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the item at the given index.
        This is a tuple of two tensors, the input and the target.
        Both will be of the same shape (length max_length)
        """
        return self.input_ids[idx], self.target_ids[idx]
