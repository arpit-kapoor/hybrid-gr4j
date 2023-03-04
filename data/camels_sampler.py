from __future__ import absolute_import

from math import ceil

from typing import Iterator, Union, Iterable, List

from torch.utils.data import Sampler, Dataset



class CamelsSampler(Sampler):

    def __init__(self, data_source: Dataset) -> None:
        super(CamelsSampler, self).__init__(data_source)
        self.data_source = data_source
        self.num_samples = len(data_source)

    def __iter__(self) -> Iterator:
        return iter(range(self.num_samples))
    
    def __len__(self):
        return self.num_samples
    



class CamelsBatchSampler(Sampler):

    def __init__(self, data_source: Dataset, 
                 window_size: int, batch_size: int, 
                 drop_last: bool=False) -> None:
        
        super(CamelsBatchSampler, self).__init__(data_source)

        self.data_source = data_source
        self.batch_size = batch_size
        self.window_size = window_size
        self.drop_last = drop_last

        self.num_samples = len(self.data_source)

        if drop_last:
            self.num_batches = int(self.num_samples/self.batch_size)
        else:
            self.num_batches = int(ceil(self.num_samples/self.batch_size))
        
    
    def __iter__(self) -> Iterator[List[int]]:
        return iter([list(range(max(0, i-self.window_size-1), min(i+self.batch_size, self.num_samples))) for i in range(0, self.num_samples, self.batch_size)])

    def __len__(self) -> int:
        return self.num_batches
    




    
