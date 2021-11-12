"""
Inspired by https://stackoverflow.com/a/50381071/14647356

Tools to create abstract classes
"""


class AbstractAttribute:
    pass


def abstract_attribute(obj=None):
    if obj is None:
        obj = AbstractAttribute()
    obj.__is_abstract_attribute__ = True
    return obj


def check_abstract_attribute_instanciation(cls):
    abstract_attributes = {
        name for name in dir(cls) if getattr(getattr(cls, name), "__is_abstract_attribute__", False)
    }
    if abstract_attributes:
        raise NotImplementedError(
            f"Can't instantiate abstract class {type(cls).__name__}."
            f"The following abstract attributes shoud be instanciated: {', '.join(abstract_attributes)}"
        )


def super_new(abstract_cls, cls, *args, **kwargs):
    __new__ = super(abstract_cls, cls).__new__
    if __new__ is object.__new__:
        return __new__(cls)
    else:
        return __new__(cls, *args, **kwargs)
