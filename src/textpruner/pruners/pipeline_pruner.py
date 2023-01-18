from .transformer_pruner import TransformerPruner
from .vocabulary_pruner import VocabularyPruner
from typing import Optional
from ..configurations import GeneralConfig,VocabularyPruningConfig,TransformerPruningConfig
import torch
from torch import nn
import os
import logging
logger = logging.getLogger(__name__)
from .utils import infer_model_type
from ..model_map import MODEL_MAP

class PipelinePruner:
    '''
    Args:
        model : The model to be pruned.
        tokenizer : The tokenizer for the model.
        vocabulary_pruning_config : a :class:`~textpruner.configurations.VocabularyPruningConfig` object.
        transformer_pruning_config : a :class:`~textpruner.configurations.TransformerPruningConfig` object.
        general_config : a :class:`~textpruner.configurations.GeneralConfig` object.
        base_model_prefix : The prefix of the base model, i.e., the name of the base model as a member in the model. \
For example, if ``model.bert_encoder = BertModel(...)``, then the ``base_model_prefix`` is ``bert_encoder``. \
TextPruner will infer the ``base_model_prefix`` so we can leave its value as ``None``. But if it fails, users have to set its value explicitly.
    '''
    def __init__(self,
                 model: nn.Module,
                 tokenizer,
                 transformer_pruning_config: Optional[TransformerPruningConfig] = None,
                 vocabulary_pruning_config : Optional[VocabularyPruningConfig] = None,
                 general_config: Optional[GeneralConfig] = None,
                 base_model_prefix : Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer

        self.general_config = GeneralConfig() if general_config is None else general_config
        self.transformer_pruning_config = TransformerPruningConfig() if transformer_pruning_config is None else transformer_pruning_config
        self.vocabulary_pruning_config = VocabularyPruningConfig() if vocabulary_pruning_config is None else vocabulary_pruning_config


        self.output_dir = self.general_config.output_dir
        base_model, model_type = infer_model_type(model, base_model_prefix)
        assert model_type in MODEL_MAP, \
            f"Model type {model_type} is not supported, or not understood. Model type must be one of {list(MODEL_MAP.keys())}"
        self.base_model = base_model
        self.model_type = model_type

        self.vocabulary_pruner = VocabularyPruner(model, tokenizer, vocabulary_pruning_config, general_config, base_model_prefix=base_model_prefix)
        self.transformer_pruner = TransformerPruner(model, transformer_pruning_config, general_config, base_model_prefix=base_model_prefix)
        self.save_dir = None

    def prune(self, 
                dataloader=None, 
                adaptor=None,
                batch_postprocessor=None,
                head_mask: Optional[torch.Tensor] =None, 
                ffn_mask: Optional[torch.Tensor]=None, 
                keep_shape=False,
                dataiter=None, 
                additional_tokens=None, 
                additional_token_ids=None, 
                save_model=True) -> Optional[str]:
        '''
        Prunes the transformers, then prunes the vocabulary.

        Args:
            dataloader : a dataloader that generates batches. Each batch should contains both the inputs and the labels.
            adaptor : a function that takes the model output and return the loss.
            batch_postprocessor : a function that takes the batch produced by the dataloader and return a batch. It is used for post-processing the batches if needed.
            head_mask : a tensor of shape ``(num_layers, num_attention_heads)``.  `1` means to keep, `0` means to prune.
            ffn_mask : a tensor of shape ``(num_layers, intermediate_hidden_size)``.  `1` means to keep, `0` means to prune.
            keep_shape : if ``True``, the model is no actually pruned and the model stucture is not changed, but the weights that *should be pruned* are set to zero.
            dataiter : a list of pre-tokenized strings. These strings will be tokenized by the tokenizer to generate a set of tokens.
            additional_tokens : a list of tokens.  These tokens must be existed in the original vocabulary.
            additional_token_ids : a list of ints representing the token ids.
            save_model : whether to save the model when the pruning is finished.
        '''

        logger.info("Transfomer pruning...")
        self.transformer_pruner.prune(dataloader, 
                                      adaptor, 
                                      batch_postprocessor=batch_postprocessor, 
                                      keep_shape=keep_shape, 
                                      head_mask=head_mask, 
                                      ffn_mask=ffn_mask, 
                                      save_model=False)
        logger.info("Vocabulary pruning...")                       
        self.vocabulary_pruner.prune(dataiter=dataiter, 
                                     additional_tokens=additional_tokens, 
                                     additional_token_ids=additional_token_ids, 
                                     save_model=False)

        if save_model is True:
            self.save_dir = self.save_model()
            return self.save_dir

    def save_model(self, dir_name=None) -> str:
        ffn_sizes = self.transformer_pruner.ffn_mask.to(int).sum(-1).tolist()
        if self.transformer_pruner.keep_shape is False:
            ffn_size = ffn_sizes[0]
            num_of_heads = self.transformer_pruner.head_mask.sum().item() / self.transformer_pruner.head_mask.size(0) 
            if len(set(ffn_sizes)) != 1:
                raise NotImplementedError("Cannot save pruned model with different ffn size per layer with keep_shape=False. \
Call PipelinePruner.save_masks or PipelinePruner.save_jit_model manually instead.")
            else:
                self.base_model.config.intermediate_size = ffn_size
        else:
            ffn_size = self.transformer_pruner.ffn_mask.size(1) #base_model.config.intermediate_size
            num_of_heads = self.transformer_pruner.head_mask.size(1) #self.transformer_pruning_config.target_num_of_heads
        
        vocab_size = len(self.vocabulary_pruner.pruned_token_ids)
        self.base_model.config.vocab_size = vocab_size


        if dir_name is None:
            save_dir = os.path.join(self.general_config.output_dir,f'pruned_V{vocab_size}H{num_of_heads}F{ffn_size}')
        else:
            save_dir = os.path.join(self.general_config.output_dir,dir_name)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(),os.path.join(save_dir,'pytorch_model.bin'))
        # save config
        self.base_model.config.save_pretrained(save_dir)
        # save tokenizer
        self.vocabulary_pruner.tokenizer_helper.save_vocab(self.tokenizer, self.vocabulary_pruner.pruned_token_ids, save_dir)

        logger.info(f"Model and configuration have been saved to {save_dir}")

        return save_dir

    def save_jit_model(self, example_inputs, dir_name=None) -> str:
        self.model.eval()
        with torch.no_grad():
            traced_model = torch.jit.trace(self.model, example_inputs=example_inputs, strict=False)
        if dir_name is None:
            save_dir = os.path.join(self.general_config.output_dir,'pruned_H{num_of_heads}F{ffn_size}_traced')
        else:
            save_dir = os.path.join(self.general_config.output_dir,dir_name)
        os.makedirs(save_dir, exist_ok=True)
        torch.jit.save(traced_model, os.path.join(save_dir,'pytorch_model.ts'))

        return save_dir