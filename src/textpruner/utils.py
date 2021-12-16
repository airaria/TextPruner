import torch
from collections.abc import Mapping
from tqdm import tqdm
import time
from typing import Tuple, Union,Dict,Optional,List

class LayerNode:
    def __init__(self,name,parent=None,value=None,fullname=None):
        self.name = name
        self.fullname = fullname
        self.value = None
        self.children_name = {}
        self.parent = parent
    def __contains__(self, key):
        return key in self.children_name
    def __getitem__(self,key):
        return self.children_name[key]
    def __setitem__(self,key,value):
        self.children_name[key]=value
    def update(self,value):
        if self.parent:
            if self.parent.value is None:
                self.parent.value = value
            else:
                if isinstance(value,(tuple,list)):
                    old_value = self.parent.value
                    new_value = [old_value[i]+value[i] for i in range(len(value))]
                    self.parent.value = new_value
                else:
                    self.parent.value += value
            if self.name.endswith('(shared)'):
                if self.parent.name.endswith('shared)'):
                    pass
                elif self.parent.value[0] == 0:
                    self.parent.name += '(shared)'
                else:
                    self.parent.name += '(partially shared)'

            self.parent.update(value)

    def format(self, level=0, total=None ,indent='--',max_level=None,max_length=None):
        string =''
        if total is None:
            total = self.value[0]
        if level ==0:
            max_length = self._max_name_length(indent,'  ',max_level=max_level) + 1
            string += '\n'
            string +=f"{'LAYER NAME':<{max_length}}\t{'#PARAMS':>15}\t{'RATIO':>10}\t{'MEM(MB)':>8}\n"

        if max_level is not None and level==max_level:
            string += f"{indent+self.name+':':<{max_length}}\t{self.value[0]:15,d}\t{self.value[0]/total:>10.2%}\t{self.value[1]:>8.2f}\n"
        else:
            if len(self.children_name)==1:
                string += f"{indent+self.name:{max_length}}\n"
            else:
                string += f"{indent+self.name+':':<{max_length}}\t{self.value[0]:15,d}\t{self.value[0]/total:>10.2%}\t{self.value[1]:>8.2f}\n"
            for child_name, child in self.children_name.items():
                string += child.format(level+1, total, 
                                       indent='  '+indent, max_level=max_level,max_length=max_length) 
        return string

    def _max_name_length(self,indent1='--', indent2='  ',level=0,max_level=None):
        length = len(self.name) + len(indent1) + level *len(indent2)
        if max_level is not None and level >= max_level:
            child_lengths = []
        else:
            child_lengths = [child._max_name_length(indent1,indent2,level=level+1,max_level=max_level) 
                            for child in self.children_name.values()]
        max_length = max(child_lengths+[length])
        return max_length


def summary(model : Union[torch.nn.Module,Dict], max_level : Optional[int] = 2):
    """
    Show the summary of model parameters.

    Args:
        model: the model to be inspected, can be a torch module or a state_dict.
        max_level: The max level to display. If ``max_level==None``, show all the levels.
    Returns:
        A formatted string.

    Example::

        print(textpruner.summay(model))

    """
    if isinstance(model,torch.nn.Module):
        state_dict = model.state_dict()
    elif isinstance(model,dict):
        state_dict = model
    else:
        raise TypeError("model should be either torch.nn.Module or a dict")
    hash_set = set()
    model_node = LayerNode('model',fullname='model')
    current = model_node
    for key,value  in state_dict.items():
        names = key.split('.')
        for i,name in enumerate(names):
            if name not in current:
                current[name] = LayerNode(name,parent=current,fullname='.'.join(names[:i+1]))
            current = current[name]

        if (value.data_ptr()) in hash_set:
            current.value = [0,0]
            current.name += "(shared)"
            current.fullname += "(shared)"
            current.update(current.value)
        else:
            hash_set.add(value.data_ptr())
            current.value = [value.numel(),value.numel() * value.element_size() / 1024 / 1024]
            current.update(current.value)

        current = model_node

    result = model_node.format(max_level=max_level)

    return result


def inference_time(model : torch.nn.Module, dummy_inputs : Union[List,Tuple,Dict], warm_up : int = 5, repetitions : int = 10):
    """
    Measure and print the inference time of the model.

    Args:
        model: the torch module to be measured.
        dummpy_inputs: the inputs to be fed into the model, can be a list ,tuple or dict.
        warm_up: Number of steps to warm up the device.
        repetitions: Number of steps to perform forward propagation. More repetitions result in more accurate measurements.

    Example::

        input_ids = torch.randint(low=0,high=10000,size=(32,256))
        textpruner.inference_time(model,dummy_inputs=[input_ids])

    """
    device = model.device
    is_train = model.training
    model.eval()

    if device.type == 'cpu':
        mean, std = cpu_inference_time(model, dummy_inputs, warm_up, repetitions)
    elif device.type == 'cuda':
        mean, std = cuda_inference_time(model, dummy_inputs, warm_up, repetitions)
    else:
        raise ValueError(f"Unknown device {device}")

    model.train(is_train)
    print(f"Device: {device}")
    print(f"Mean inference time: {mean:.2f}ms")
    print(f"Standard deviation: {std:.2f}ms")

    return mean, std


def cuda_inference_time(model : torch.nn.Module, dummy_inputs, warm_up, repetitions):
    device = model.device
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings=torch.zeros(repetitions)
    with torch.no_grad():
        for _ in tqdm(range(warm_up),desc='cuda-warm-up'):
            if isinstance(dummy_inputs, Mapping):
                inputs = {k: v.to(device) for k,v in dummy_inputs.items()}
                _ = model(**inputs)
            else:
                inputs = [t.to(device) for t in dummy_inputs]
                _ = model(*inputs)
        for rep in tqdm(range(repetitions),desc='cuda-repetitions'):
            if isinstance(dummy_inputs, Mapping):
                inputs = {k: v.to(device) for k,v in dummy_inputs.items()}
                starter.record()
                _ = model(**inputs)
                ender.record()
            else:
                inputs = [t.to(device) for t in dummy_inputs]
                starter.record()
                _ = model(*inputs)
                ender.record()
            torch.cuda.synchronize()
            elapsed_time_ms = starter.elapsed_time(ender)
            timings[rep] = elapsed_time_ms
    mean = timings.sum().item() / repetitions
    std = timings.std().item()

    return mean, std


def cpu_inference_time(model : torch.nn.Module, dummy_inputs, warm_up, repetitions):
    device = model.device
    timings=torch.zeros(repetitions)
    with torch.no_grad():
        for _ in tqdm(range(warm_up),desc='cpu-warm-up'):
            if isinstance(dummy_inputs, Mapping):
                inputs = {k: v.to(device) for k,v in dummy_inputs.items()}
                _ = model(**inputs)
            else:
                inputs = [t.to(device) for t in dummy_inputs]
                _ = model(*inputs)
        for rep in tqdm(range(repetitions),desc='cpu-repetitions'):
            if isinstance(dummy_inputs, Mapping):
                inputs = {k: v.to(device) for k,v in dummy_inputs.items()}
                start = time.time()
                _ = model(**inputs)
                end = time.time()
            else:
                inputs = [t.to(device) for t in dummy_inputs]
                start = time.time()
                _ = model(*inputs)
                end = time.time()
            elapsed_time_ms = (end - start) * 1000
            timings[rep] = elapsed_time_ms
    mean = timings.sum().item() / repetitions
    std = timings.std().item()

    return mean, std