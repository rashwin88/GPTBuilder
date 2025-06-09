"""
Uses the GPT data prep V1 class to create a dataloader.
"""

from data_preparation.gpt_data_prep_v1 import GPTDataPrepV1
from torch.utils.data import DataLoader
from typing import Tuple
from namespaces.test_input_types import TextInputType
import tiktoken


def create_data_loader(
    text_input_type: TextInputType,
    text_input: str,
    max_length: int,
    stride: int,
    num_workers: int,
    batch_size: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Creates a dataloader from the GPT data prep V1 class. Uses the gpt2 tokenizer.

    Args:
        text_input_type (TextInputType): The type of text input that is provided.
        text_input (str): The text input to be processed.
            If the TextInputType is FILE_PATH, then the text input is a path to a file.
            If the TextInputType is STRING, then the text input is a string.
            Passed on to the GPTDataPrepV1 class.
        max_length (int): The maximum length of the text input.
            Passed on to the GPTDataPrepV1 class.
        stride (int): The stride of the sliding window.
            Passed on to the GPTDataPrepV1 class.
        num_workers (int): The number of workers to be used.
        batch_size (int): The batch size to be used.
        shuffle (bool): Whether to shuffle the data.
        drop_last (bool): Whether to drop the last batch if it is not of the same size as the batch size.

    Returns:
        DataLoader: A dataloader from the GPT data prep V1 class.
    """

    # Create the tokenizer. Here we specifically use the gpt2 tokenizer.
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create the dataset.
    dataset = GPTDataPrepV1(
        text_input_type=text_input_type,
        text_input=text_input,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    return dataloader
