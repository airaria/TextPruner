import torch
from torch import nn
from collections import abc
from typing import Tuple,Optional
import logging
logger = logging.getLogger(__name__)

def move_to_device(batch, device):
    r"""Puts each data field to the device"""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch,(list,tuple)):
        return tuple(move_to_device(item,device) for item in batch)
    elif isinstance(batch, abc.Mapping):
        return {key: move_to_device(value,device) for key, value in batch.items()}
    else:
        return batch


def infer_model_type(model, base_model_prefix) -> Tuple[nn.Module,str]:
        if base_model_prefix is not None:
            base_model = getattr(model, base_model_prefix, model)
            model_type = base_model.config.model_type
        else:
            if hasattr(model, 'base_model_prefix'):
                base_model = getattr(model, model.base_model_prefix, model)
                if hasattr(base_model, 'config'):
                    model_type = base_model.config.model_type
                else:
                    raise ValueError("Cannot get model_type! You should provide base_model_prefix")
            else:
                raise ValueError("Cannot get model_type! You should provide base_model_prefix")
        return base_model, model_type


def random_mask_tensor(shape: Tuple[int,int], p : float = 0.5, dtype=None, even_masks=True):
    tensor = torch.zeros(shape)
    if even_masks is False:
        tensor = tensor.bernoulli_(p=0.5)
    else:
        num_masks_per_row = int(shape[1] * p)
        for i in range(shape[0]):
            tensor[i][:num_masks_per_row] = 1
            randindex = torch.randperm(shape[1])
            tensor[i] = tensor[i][randindex]
    if dtype is not None:
        return tensor.to(dtype)
    else:
        return tensor


def generate_mask(importance : torch.Tensor, total_target_size : int, even_masking : bool = False, 
                layer_start : Optional[int] = None, layer_end: Optional[int] = None, multiple_of : int = 1 ) -> torch.Tensor:
    if layer_start is not None and layer_end is not None:
        target_size_per_layer = total_target_size // importance.size(0)
        mask = torch.ones_like(importance)
        for i in range(layer_start,layer_end):
            layer = importance[i]
            importance_layer_order = torch.argsort(layer)
            mask[i][importance_layer_order[:-target_size_per_layer]] = 0
    elif even_masking is True:
        target_size_per_layer = total_target_size // importance.size(0)
        mask = torch.ones_like(importance)
        for i,layer in enumerate(importance):
            importance_layer_order = torch.argsort(layer)
            mask[i][importance_layer_order[:-target_size_per_layer]] = 0
    elif multiple_of == 1:
        importance_flat = importance.reshape(-1)
        importance_order = torch.argsort(importance_flat)   # ascending
        mask_flat = torch.ones_like(importance_flat)
        for pos in importance_order[:-total_target_size]:
            mask_flat[pos] = 0
        mask = mask_flat.reshape(importance.shape)
    else:
        num_layers = importance.size(0)
        num_groups = importance.size(1) // multiple_of
        importance_order_2d = torch.argsort(importance,dim=-1)
        importance_3d = torch.zeros(num_layers, num_groups, multiple_of).to(importance)
        for i, layer_order in enumerate(importance_order_2d):
            layer_sorted_by_importance = importance[i][layer_order].view(-1,multiple_of) # (num_head // multiple_of, multiple_of)
            importance_3d[i] = layer_sorted_by_importance
        importance_2d_order_2d = importance_order_2d.view(num_layers * num_groups, multiple_of)

        importance_3d_s_flat = importance_3d.sum(-1).view(-1) # num_layers * num_groups
        importance_3d_s_flat_order_flat = torch.argsort(importance_3d_s_flat)   # ascending

        total_group_target_size = total_target_size // multiple_of
        mask = torch.ones_like(importance)

        for pos in importance_3d_s_flat_order_flat[:-total_group_target_size]:
            x = int(pos) // num_groups
            mask[x,importance_2d_order_2d[pos]] = 0

    # check for disconnected graph
    mask_sum = mask.sum(-1)
    for i in range(len(mask_sum)):
        if mask_sum[i]==0:
            print("Warning")
            most_imp = torch.argmax(importance[i])
            mask[i][most_imp] = 1
    return mask


def infer_logits(outputs,adaptor=None):
    if adaptor is None:
        try:
            if isinstance(outputs, torch.Tensor):
                logits = outputs
                assert len(logits.size())>0
            elif isinstance(outputs, (list,tuple)):
                logits = outputs[0]
                assert len(logits.size())>0
            elif isinstance(outputs, abc.Mapping):
                logits = outputs['logits']
            else:
                logits = outputs.logits
        except (KeyError, AttributeError, AssertionError) as e:
            logger.error("Cannot infer logits from the outputs automatically! An adaptor is needed")
            raise e
    else:
        logits = adaptor(outputs)
    return logits


def infer_loss(outputs, adaptor=None):
    if adaptor is None:
        try:
            if isinstance(outputs, torch.Tensor):
                loss = outputs
                assert len(loss.size())==0
            elif isinstance(outputs, (list,tuple)):
                loss = outputs[0]
                assert len(loss.size())==0
            elif isinstance(outputs, abc.Mapping):
                loss = outputs['loss']
            else:
                loss = outputs.loss
        except (KeyError, AttributeError, AssertionError) as e:
            logger.error("Cannot infer loss from the outputs automatically! An adaptor is needed")
            raise e
    else:
        loss = adaptor(outputs)
    return loss