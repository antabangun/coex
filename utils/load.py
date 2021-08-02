import importlib


def make_list(var, n=None):
    var = var if isinstance(var, list) else [var]
    if n is None:
        return var
    else:
        assert len(var) == 1 or len(var) == n
        return var * n if len(var) == 1 else var


def load_class(filename, paths, concat=True):
    for path in make_list(paths):
        full_path = '{}.{}'.format(path, filename) if concat else path
        if importlib.util.find_spec(full_path):
            return getattr(importlib.import_module(full_path), filename)
    raise ValueError('Unknown class {}'.format(filename))
