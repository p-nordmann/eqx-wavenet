from typing import NamedTuple


class WavenetConfig(NamedTuple):
    # High-level architecture
    num_layers: int
    layer_dilations: list[int]

    # Input
    size_in: int
    input_kernel_size: int

    # Layers
    size_layers: int

    # Head
    size_hidden: int
    size_out: int
