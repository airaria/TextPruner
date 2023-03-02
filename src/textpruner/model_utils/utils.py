import re
import torch
from torch import nn
import logging
from typing import Dict
logger = logging.getLogger(__name__)

class DefaultModelVocabResizer:
    @classmethod
    def set_embeddings(cls, model, token_ids):
        # self.model.get_input_embeddings()
        old_word_embeddings = model.embeddings.word_embeddings
        old_word_embeddings_weight = old_word_embeddings.weight

        pruned_word_embeddings_weight = torch.index_select(
            old_word_embeddings_weight, 0, index=torch.LongTensor(token_ids).to(old_word_embeddings_weight.device))
        pruned_num_tokens, embedding_dim = pruned_word_embeddings_weight.shape

        pruned_word_embeddings = nn.Embedding(
            pruned_num_tokens, embedding_dim).to(old_word_embeddings_weight.device)
        pruned_word_embeddings.weight.data[:] = pruned_word_embeddings_weight[:]

        model.embeddings.word_embeddings = pruned_word_embeddings

    @classmethod
    def set_lm_head(cls, model, token_ids) -> bool:
        try:
            output_embedding_layer = model.get_output_embeddings()
        except AttributeError:
            return False
        if output_embedding_layer is None:
            return False
        output_embedding_layer.weight = model.get_input_embeddings().weight
        if hasattr(output_embedding_layer,'bias') and output_embedding_layer.bias is not None:
            output_embedding_layer.bias.data = torch.index_select(
                output_embedding_layer.bias.data, 0, index=torch.LongTensor(token_ids).to(output_embedding_layer.weight.device))
        return True


#bert, roberta, xlmr, ...
def get_word_embeddings(model):
    state_dict = model.state_dict()
    layer_template = "embeddings.word_embeddings"
    layer_names = []
    for key in state_dict:
        if layer_template in key:
            layer_names.append(key)
    assert len(
        layer_names) == 1, f"Invalid model structure with ambiguous word embeddings: {layer_names}"
    word_embedding_weight = state_dict[layer_names[0]]
    return word_embedding_weight


#bert, roberta, xlmr, ...
def get_num_of_trms(model):
    layer_template_regex = "encoder.layer\.(\d+)\."
    layer_template = "encoder.layer.LAYER_INDEX."
    layer_indices = set()
    layer_names = set()
    state_dict = model.state_dict()
    for key in state_dict:
        matched = re.findall(layer_template_regex, key)
        if len(matched) > 0:
            assert len(
                matched) == 1, f"Invalid model structure. Cannot parse {key}"
            layer_index = int(matched[0])
            layer_indices.add(layer_index)

            layer_name = layer_template.replace("LAYER_INDEX", matched[0])
            layer_name = key[:key.find(layer_name)]+layer_name
            layer_names.add(layer_name)

    print("Found transfomr layers:", layer_indices)
    print("Layer name prefixes:", layer_names)

    return len(layer_indices), layer_names
