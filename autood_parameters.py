from dataclasses import dataclass
from typing import List


@dataclass
class AutoODDefaultParameters:
    dataset: str
    k_range: List[int]
    if_range: List[float]
    N_range: List[int]


autood_datasets_to_parameters = {
    "pageblocks": AutoODDefaultParameters(
        dataset="PageBlocks",
        k_range=list(range(10, 110, 10)),
        if_range=[0.5, 0.6, 0.7, 0.8, 0.9],
        N_range=[0.05, 0.07, 0.09, 0.11, 0.13, 0.15]
    )
}


def get_default_parameters(dataset):
    if dataset in autood_datasets_to_parameters:
        return autood_datasets_to_parameters[dataset]
    return AutoODDefaultParameters(
        dataset=dataset,
        k_range=list(range(10, 110, 10)),
        if_range=[0.5, 0.6, 0.7, 0.8, 0.9],
        N_range=[0.05, 0.07, 0.09, 0.11, 0.13, 0.15]
    )
