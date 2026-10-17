class ModelError(Exception):
    pass


class ConfigurationError(ModelError):
    pass


class ModelLoadError(ModelError):
    pass


class ModelNotLoadedError(ModelError):
    pass


class InvalidFrameError(ModelError):
    pass
