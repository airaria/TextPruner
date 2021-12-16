from torch.utils import data
from ..pruners import VocabularyPruner, TransformerPruner, PipelinePruner
from ..utils import summary
from .utils import read_file_line_by_line
import logging
logger = logging.getLogger(__name__)

def call_vocabulary_pruning(configurations, model, tokenizer, vocabulary_file):
    general_config = configurations["GeneralConfig"]
    vocabulary_pruning_config = configurations["VocabularyPruningConfig"]
    pruner = VocabularyPruner(model, tokenizer, vocabulary_pruning_config, general_config)
    texts,is_token_ids = read_file_line_by_line(vocabulary_file)
    if is_token_ids is False:
        output_dir = pruner.prune(dataiter=texts, save_model=True)
    else:
        output_dir = pruner.prune(additional_token_ids=texts, save_model=True)

    print("After pruning:")
    print(summary(model))


def call_transformer_pruning(configurations, model, dataloader, adaptor):
    general_config = configurations["GeneralConfig"]
    transformer_pruning_config = configurations["TransformerPruningConfig"]
    pruner = TransformerPruner(model, transformer_pruning_config, general_config)

    keep_shape = False
    if transformer_pruning_config.ffn_even_masking is False:
        logger.warning("ffn_even_masking is False. Cannot save pruned model with different ffn size. \
A full model with the relevant weights set to zero will be saved. \
You can save a pruned TorchScript model, \
use the textpruner.TransformerPruner.save_jit_model in your python script.")
        keep_shape = True
    output_dir = pruner.prune(dataloader=dataloader, adaptor=adaptor, keep_shape=keep_shape, save_model=True)       
    print("After pruning:")
    print(summary(model))


def call_pipeling_pruning(configurations, model, tokenizer, vocabulary_file, dataloader, adaptor):
    general_config = configurations["GeneralConfig"]
    vocabulary_pruning_config = configurations["VocabularyPruningConfig"]
    transformer_pruning_config = configurations["TransformerPruningConfig"]
    pruner = PipelinePruner(model, tokenizer, 
                            transformer_pruning_config, 
                            vocabulary_pruning_config,
                            general_config)
    texts,is_token_ids = read_file_line_by_line(vocabulary_file)
    keep_shape = False
    if transformer_pruning_config.ffn_even_masking is False:
        logger.warning("ffn_even_masking is False. Cannot save pruned model with different ffn size. \
A full model with the relevant weights set to zero will be saved. \
You can save a pruned TorchScript model, \
use the textpruner.TransformerPruner.save_jit_model in your python script.")
        keep_shape = True
    if is_token_ids is False:
        output_dir = pruner.prune(dataloader=dataloader, adaptor=adaptor, dataiter=texts, keep_shape=keep_shape, save_model=True)
    else:
        output_dir = pruner.prune(dataloader=dataloader, adaptor=adaptor, additional_token_ids=texts, keep_shape=keep_shape, save_model=True)

    print("After pruning:")
    print(summary(model))