"""Fail closed if a cached neural epoch attempts a real physical operator."""
import contextvars
from contextlib import contextmanager
_EPOCH=contextvars.ContextVar('a18_cached_only_epoch',default=None)

def assert_operator_allowed(name):
    record=_EPOCH.get()
    if record is not None:
        record['attempted_Maxwell_operations'].append(name)
        raise RuntimeError('Maxwell operator forbidden in cached epoch: '+name)

@contextmanager
def cached_only_epoch():
    record={'Maxwell_action_count':0,'attempted_Maxwell_operations':[]};token=_EPOCH.set(record)
    try:
        yield record
        if record['attempted_Maxwell_operations']:raise RuntimeError('cached epoch violated isolation')
    finally:_EPOCH.reset(token)
