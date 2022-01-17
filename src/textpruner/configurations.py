from dataclasses import asdict
import torch
import json
import logging
from typing import Union, Optional
from dataclasses import dataclass, asdict
logger = logging.getLogger(__name__)



@dataclass
class Config:
    """Base class for :class:`~textpruner.configurations.GeneralConfig`, 
    :class:`~textpruner.configurations.VocabularyPruningConfig` and :class:`~textpruner.configurations.TransformerPruningConfig`."""

    @classmethod
    def from_json(cls, json_filename: str):
        """Construct the configuration from a json file."""
        with open(json_filename,'r') as f:
            config_map = json.load(f)
        config = CONFIG_CLASS[config_map['config_class']].from_dict(config_map)
        return config

    @classmethod
    def from_dict(cls, config_map: dict):
        """Construct the configuration from a dict."""
        config = CONFIG_CLASS[config_map['config_class']](**config_map)
        return config


    def save_to_json(self, json_filename: str):
        """Save the configuration the a json file."""
        config_map = asdict(self)
        with open(json_filename,'w') as f:
            json.dump(config_map, f, indent = 2)


@dataclass
class GeneralConfig(Config):

    '''
    Configurations for the device and the output directory.

    Args:
        device: ``'cpu'`` or ``'cuda'`` or ``'cuda:0'`` etc. Specify which device to use. If it is set to ``'auto'``, 
            TextPruner will try to use the CUDA device if there is one; otherwise uses CPU.
        output_dir: The diretory to save the pruned models.
        config_class: Type of the configurations. Users should not change its value.
    '''
    use_device: str = 'auto'
    output_dir: str = './pruned_models'
    config_class : str = "GeneralConfig"
    def __post_init__(self):
        if self.use_device == 'auto':
            if torch.cuda.is_available():
                logger.info(f"Using current cuda device")
                self.device = ('cuda')
            else:
                logger.info(f"Using cpu device")
                self.device = ('cpu')
        else:
            self.device = self.use_device

@dataclass
class VocabularyPruningConfig(Config):
    '''
    Configurations for vocabulary pruning.

    Args:
        min_count: The threshold to decide if the token should be removed.
            The token will be removed from the vocabulary if it appears less than ``min_count`` times in the corpus.
        prune_lm_head: whether pruning the lm_head if the model has one. If ``prune_lm_head==False``, TextPruner will not prune the lm_head; 
            if ``prune_lm_head==True``, TextPruner will prune the lm_head and raise a error if the model does not have an lm_head;
            if ``prune_lm_head=='auto'``, TextPruner will try to prune the lm_head and will continue if the model does not have an lm_head.
        config_class: Type of the configurations. Users should not change its value.
    '''
    min_count: int = 1
    prune_lm_head : Union[bool,str] = 'auto'
    config_class: str = "VocabularyPruningConfig"


@dataclass
class TransformerPruningConfig(Config):
    """
    Configurations for transformer pruning.

    Args:
        target_ffn_size : the target average FFN size per layer.
        target_num_of_heads : the target average number of heads per layer.
        pruning_method : ``'masks'`` or ``'iterative'``. If set to ``'masks'``, the pruner prunes the model with the given masks (``head_mask`` and ``ffn_mask``).
                        If set to ``'iterative'``. the pruner calculates the importance scores of the neurons based on the data provided by the ``dataloader`` and then prunes the model based on the scores.
        ffn_even_masking : Whether the FFN size of each layer should be the same. 
        head_even_masking : Whether the number of attention heads of each layer should be the same.
        n_iters :  if ``pruning_method`` is set to ``'iterative'``, ``n_iters`` is number of pruning iterations to prune the model progressively.
        multiple_of : if ``ffn_even_masking`` is ``False``, restrict the target FFN size of each layer to be a multiple of ``multiple_if``.
        pruning_order: ``None`` or ``'head-first'`` or ``'ffn-first'``. ``None``: prune the attention heads and ffn layer simultaneously; if set to ``'head-first'`` or ``'ffn-first'``, the actual number of iterations is ``2*n_iters``.
        use_logits : if ``True``, performs self-supervised pruning, where the logits are treated as the soft labels.
        config_class: Type of the configurations. Users should not change its value.

    Warning:
        if ``ffn_even_masking`` is ``False``, the pruned model can not be save normally (we cannot load the model with the transformers libarary with the saved weights).
        So make sure to set ``save_model=False`` when calling ``TransformerPruner.prune()`` or ``PipelinePruner.prune()``. 
        There are two ways to avoid this:
        
        * Save the model in TorchScript format manually;
        * Set ``keep_shape=False`` when calling ``TransformerPruner.prune()`` or ``PipelinePruner.prune()``, so the full model can be saved. Then save the ``ffn_masks`` and ``head_masks``. When loading the model, load the full model and then prune it with the masks.
    """

    target_ffn_size : Optional[int] = None
    target_num_of_heads: Optional[int] = None
    pruning_method : str = 'masks'
    ffn_even_masking : Optional[bool] = True
    head_even_masking : Optional[bool] = True
    n_iters : Optional[int] = 1
    multiple_of : int = 1
    pruning_order : Optional[str] = None
    use_logits : bool = False
    config_class: str = "TransformerPruningConfig"
    def __post_init__(self):
        assert self.pruning_method in ('masks','iterative'), "Unrecgonized pruning method"
        assert (self.pruning_order is None) or (self.pruning_order in ('head-first','ffn-first')), "Unrecgonized pruning order"
        if self.ffn_even_masking is False:
            logger.warning("ffn_even_masking is False. Pruned model can only be save in TorchScript format manually.")

CONFIG_CLASS = {
    'GeneralConfig': GeneralConfig,
    'VocabularyPruningConfig': VocabularyPruningConfig,
    'TransformerPruningConfig': TransformerPruningConfig
}