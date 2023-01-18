import torch
from torch import nn
import os
from ..model_map import MODEL_MAP
from ..configurations import VocabularyPruningConfig, GeneralConfig
from .utils import infer_model_type
import logging
from tqdm import tqdm
from collections import abc
from typing import Optional
logger = logging.getLogger(__name__)

class VocabularyPruner:
    """
    Args:
        model : The model to be pruned.
        tokenizer : The tokenizer for the model.
        vocabulary_pruning_config : a :class:`~textpruner.configurations.VocabularyPruningConfig` object.
        general_config : a :class:`~textpruner.configurations.GeneralConfig` object.
        base_model_prefix : The prefix of the base model, i.e., the name of the base model as a member in the model. \
For example, if ``model.bert_encoder = BertModel(...)``, then the ``base_model_prefix`` is ``bert_encoder``. \
TextPruner will infer the ``base_model_prefix`` so we can leave its value as ``None``. But if it fails, users have to set its value explicitly.

    """
    def __init__(self,
                 model : nn.Module,
                 tokenizer,
                 vocabulary_pruning_config : Optional[VocabularyPruningConfig] = None,
                 general_config : Optional[GeneralConfig] = None,
                 base_model_prefix : Optional[str] = None):

        self.model = model
        self.tokenizer = tokenizer

        #infer model type
        base_model, model_type = infer_model_type(model, base_model_prefix)
        assert model_type in MODEL_MAP, \
            f"Model type {model_type} is not supported, or not understood. Model type must be one of {list(MODEL_MAP.keys())}"
        self.base_model = base_model
        self.model_type = model_type


        self.general_config = GeneralConfig() if general_config is None else general_config
        self.vocabulary_pruning_config = VocabularyPruningConfig() if vocabulary_pruning_config is None else vocabulary_pruning_config

        self.model.to(self.general_config.device)

        self.model_vocab_resizer = MODEL_MAP[self.model_type]['resizer']
        self.tokenizer_helper = MODEL_MAP[self.model_type]['tokenizer_helper']
        self.pruned_token_ids = []
        os.makedirs(self.general_config.output_dir, exist_ok=True)
        self.save_dir = None

    def prune(self, dataiter=None, additional_tokens=None, 
                               additional_token_ids=None, save_model=True) -> Optional[str]:
        '''
        Prunes the vocabulay of the model and the tokenizer. The pruner will only keep the tokens in ``dataiter``, ``additional_tokens`` and ``additional_token_ids``.

        * Use ``dataiter`` to generate a set of tokens from the raw texts.
        * Use ``additional_tokens`` or ``additional_token_ids`` to specify the tokens or token_ids directly without running the tokenization.

        Args:
            dataiter : a list of pre-tokenized strings. These strings will be tokenized by the tokenizer to generate a set of tokens.
            additional_tokens : a list of tokens.  These tokens must be existed in the original vocabulary.
            additional_token_ids : a list of ints representing the token ids.
            save_model : whether to save the model when the pruning is finished.
        '''
        min_count = self.vocabulary_pruning_config.min_count
        lm_head_pruning= self.vocabulary_pruning_config.prune_lm_head
        pruned_token_ids = self.tokenizer_helper.get_token_ids(tokenizer=self.tokenizer,
                                                            dataiter=dataiter,
                                                            additional_tokens=additional_tokens,
                                                            additional_token_ids=additional_token_ids,
                                                            min_count=min_count)
        self.model_vocab_resizer.set_embeddings(model=self.base_model, token_ids=pruned_token_ids)

        if lm_head_pruning == 'auto' or lm_head_pruning is True:
            is_success = self.model_vocab_resizer.set_lm_head(self.model, pruned_token_ids)
            if is_success is False:
                if lm_head_pruning is True:
                    logger.info("Cannot get output embeddings! Is your model has a MLM prediction head?")
                else:
                    logger.info("Cannot get output embeddings. No LM head pruning.")
        self.pruned_token_ids = pruned_token_ids

        if save_model is True:
            self.save_dir = self.save_model()
            return self.save_dir


    def save_model(self, dir_name = None) -> str:
        
        if self.model_type.lower() in ['t5', 'mt5']:
            vocab_size = self.base_model.shared.weight.shape[0]
        else:
            vocab_size = len(self.pruned_token_ids)
        self.base_model.config.vocab_size = vocab_size

        if dir_name is None:
            save_dir = os.path.join(self.general_config.output_dir, f'pruned_V{vocab_size}')
        else:
            save_dir = os.path.join(self.general_config.output_dir, dir_name)
        os.makedirs(save_dir, exist_ok=True)

        # save tokenizer
        self.tokenizer_helper.save_vocab(self.tokenizer, self.pruned_token_ids, save_dir)

        # save weights
        torch.save(self.model.state_dict(),os.path.join(save_dir,f'pytorch_model.bin'))

        # save config
        config_dir = os.path.join(save_dir)
        self.base_model.config.save_pretrained(config_dir)
        logger.info(f"Model and configuration have been saved to {save_dir}")

        return save_dir
