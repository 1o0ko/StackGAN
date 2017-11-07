import types

class DataStore(list):
    '''
    This class is used to register and instantiate datasets
    '''
    def __init__(self):
        self._default = None

    def register(self, cls, default):
        if default:
            self._default = cls

        self.append(cls)

    def create(self, datadir, cfg):
        if hasattr(cfg, 'DATASET_TYPE') and cfg.DATASET_TYPE in self:
            clazz_ = cfg.DATASET_TYPE
        else:
            clazz_ = self._default if self._default else self[0]

        return clazz_(datadir, cfg.EMBEDDING_FILENAME, cfg.HR_LR_RATIO)


#TODO: have a better way to handle the singleton pattern
datastore = DataStore()


def register(default):
    if isinstance(default, (type, types.ClassType)):
        cls = default
        datastore.register(cls, False)
        return cls
    else:
        def wrapper(cls):
            datastore.register(cls, default)
            return cls

        return wrapper