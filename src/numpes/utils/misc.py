"""Module for miscellaneous utility functions"""

def as_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, range)):
        return list(value)
    return [value]
