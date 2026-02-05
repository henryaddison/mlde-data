_ACTIONS = {}


def register_action(cls=None, *, name=None):
    """A decorator for registering action classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _ACTIONS:
            raise ValueError(f"Already registered action with name: {local_name}")

        _ACTIONS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_action(name):
    return _ACTIONS[name]
