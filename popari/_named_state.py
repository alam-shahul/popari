import hashlib
import re

import numpy as np
import torch
from torch import nn


def get_key(prefix: str, name: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_") or "unnamed"
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{slug}_{digest}"


class BufferDict(nn.ParameterDict):
    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix

    def __getitem__(self, key: str):
        if key in self:
            return super().__getitem__(key)
        return super().__getitem__(get_key(self.prefix, key))

    def __setitem__(self, key: str, value) -> None:
        key = get_key(self.prefix, key)
        if not isinstance(value, nn.Parameter):
            value = nn.Parameter(value, requires_grad=False)
        else:
            value.requires_grad_(False)
        super().__setitem__(key, value)


class ParameterDict(nn.ParameterDict):
    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix

    def __getitem__(self, key: str):
        if key in self:
            return super().__getitem__(key)
        return super().__getitem__(get_key(self.prefix, key))

    def __setitem__(self, key: str, value) -> None:
        super().__setitem__(get_key(self.prefix, key), value)
