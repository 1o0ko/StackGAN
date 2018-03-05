from .datasets import BaseDataset
from .registry import register


@register
class StreamingDataset(BaseDataset):
    '''
    HDF5 dataset implementation
    # TODO: Finish this
    '''
    def __init__(self, *args, **kwargs):
        super(StreamingDataset, self).__init__(*args, **kwargs)
