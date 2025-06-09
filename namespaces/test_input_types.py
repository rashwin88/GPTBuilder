from enum import Enum


class TextInputType(Enum):
    """
    The type of text input that is provided. The idea is that different sources of text input can be provided.
    What we need is some way of enumerating these different sources so that we can use the right methods internally to
    load data.
    """

    FILE_PATH = "file_path"
    STRING = "string"
