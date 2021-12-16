import re
import torch
from torch import nn
import logging
from typing import Dict, List
logger = logging.getLogger(__name__)

# adapted from huggingface/nn_pruning/model_structure.py
class ModelStructure:
    MODEL_PREFIX: str = ""
    ENCODER_PREFIX: str = ""
    ATTENTION_LAYERS = ("query", "key", "value")
    FFN_LAYERS = ("interm_dense", "output_dense")

    @classmethod
    def get_att_query(cls, model, ignore_model_prefix=False):
        pattern = cls.ENCODER_PREFIX + cls.LAYER_PATTERNS['query']
        if ignore_model_prefix is False:
            pattern = cls.MODEL_PREFIX + pattern
        rs = []
        for k in model.named_modules():
            name = k[0]
            r = re.search(pattern, name)
            if r is not None:
                rs.append(get_submodule(model,r.group()))
        return rs


    @classmethod
    def get_att_key(cls, model, ignore_model_prefix=False):
        pattern = cls.ENCODER_PREFIX + cls.LAYER_PATTERNS['key']
        if ignore_model_prefix is False:
            pattern = cls.MODEL_PREFIX + pattern
        rs = []
        for k in model.named_modules():
            name = k[0]
            r = re.search(pattern, name)
            if r is not None:
                rs.append(get_submodule(model,r.group()))
        return rs


    @classmethod
    def get_att_value(cls, model, ignore_model_prefix=False):
        pattern = cls.ENCODER_PREFIX + cls.LAYER_PATTERNS['value']
        if ignore_model_prefix is False:
            pattern = cls.MODEL_PREFIX + pattern
        rs = []
        for k in model.named_modules():
            name = k[0]
            r = re.search(pattern, name)
            if r is not None:
                rs.append(get_submodule(model,r.group()))
        return rs


    @classmethod
    def get_att_output(cls, model, ignore_model_prefix=False):
        pattern = cls.ENCODER_PREFIX + cls.LAYER_PATTERNS['att_dense']
        if ignore_model_prefix is False:
            pattern = cls.MODEL_PREFIX + pattern
        rs = []
        for k in model.named_modules():
            name = k[0]
            r = re.search(pattern, name)
            if r is not None:
                rs.append(get_submodule(model,r.group()))
        return rs
    

    @classmethod
    def get_ffn_interm(cls, model, ignore_model_prefix=False):
        pattern = cls.ENCODER_PREFIX + cls.LAYER_PATTERNS['interm_dense']
        if ignore_model_prefix is False:
            pattern = cls.MODEL_PREFIX + pattern
        rs = []
        for k in model.named_modules():
            name = k[0]
            r = re.search(pattern, name)
            if r is not None:
                rs.append(get_submodule(model,r.group()))
        return rs


    @classmethod
    def get_ffn_output(cls, model, ignore_model_prefix=False):
        pattern = cls.ENCODER_PREFIX + cls.LAYER_PATTERNS['output_dense']
        if ignore_model_prefix is False:
            pattern = cls.MODEL_PREFIX + pattern
        rs = []
        for k in model.named_modules():
            name = k[0]
            r = re.search(pattern, name)
            if r is not None:
                rs.append(get_submodule(model,r.group()))
        return rs

    @classmethod
    def get_num_layers(cls, model, ignore_model_prefix=False):
        pattern = cls.ENCODER_PREFIX
        if ignore_model_prefix is False:
            pattern = cls.MODEL_PREFIX + pattern
        rs = []
        for k in model.named_modules():
            name = k[0]
            r = re.search(pattern, name)
            if r is not None:
                rs.append(r.group())
        return len(set(rs))

    @classmethod
    def layer_index(cls, child_module_name):
        extracts = re.findall(r"[0-9]+", child_module_name)
        return int(extracts[0])



# from PyTorch 1.9.0
def get_submodule(model: nn.Module, target: str) -> nn.Module:
    """
    Returns the submodule given by ``target`` if it exists,
    otherwise throws an error.

    For example, let's say you have an ``nn.Module`` ``A`` that
    looks like this:

    .. code-block::text

        A(
            (net_b): Module(
                (net_c): Module(
                    (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                )
                (linear): Linear(in_features=100, out_features=200, bias=True)
            )
        )

    (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
    submodule ``net_b``, which itself has two submodules ``net_c``
    and ``linear``. ``net_c`` then has a submodule ``conv``.)

    To check whether or not we have the ``linear`` submodule, we
    would call ``get_submodule("net_b.linear")``. To check whether
    we have the ``conv`` submodule, we would call
    ``get_submodule("net_b.net_c.conv")``.

    The runtime of ``get_submodule`` is bounded by the degree
    of module nesting in ``target``. A query against
    ``named_modules`` achieves the same result, but it is O(N) in
    the number of transitive modules. So, for a simple check to see
    if some submodule exists, ``get_submodule`` should always be
    used.

    Args:
        target: The fully-qualified string name of the submodule
            to look for. (See above example for how to specify a
            fully-qualified string.)

    Returns:
        torch.nn.Module: The submodule referenced by ``target``

    Raises:
        AttributeError: If the target string references an invalid
            path or resolves to something that is not an
            ``nn.Module``
    """
    if target == "":
        return model

    atoms: List[str] = target.split(".")
    mod: torch.nn.Module = model

    for item in atoms:

        if not hasattr(mod, item):
            raise AttributeError(mod._get_name() + " has no "
                                    "attribute `" + item + "`")

        mod = getattr(mod, item)

        if not isinstance(mod, torch.nn.Module):
            raise AttributeError("`" + item + "` is not "
                                    "an nn.Module")

    return mod