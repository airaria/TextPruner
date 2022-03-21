from .utils import DefaultModelVocabResizer
from .model_structure import ModelStructure
import torch
from torch import nn 
class XLMVocabResizer(DefaultModelVocabResizer):
    model_name : str = 'xlm'

    @classmethod
    def set_embeddings(cls, model, token_ids):
        # self.model.get_input_embeddings()

        if hasattr(model.embeddings, 'word_embeddings'): #XLM
            old_word_embeddings = model.embeddings.word_embeddings
        else:
            old_word_embeddings = model.embeddings

        

        # old_word_embeddings = model.embeddings.word_embeddings
        old_word_embeddings_weight = old_word_embeddings.weight

        pruned_word_embeddings_weight = torch.index_select(
            old_word_embeddings_weight, 0, index=torch.LongTensor(token_ids).to(old_word_embeddings_weight.device))
        pruned_num_tokens, embedding_dim = pruned_word_embeddings_weight.shape

        pruned_word_embeddings = nn.Embedding(
            pruned_num_tokens, embedding_dim).to(old_word_embeddings_weight.device)
        pruned_word_embeddings.weight.data[:] = pruned_word_embeddings_weight[:]


        if hasattr(model.embeddings, 'word_embeddings'):
            model.embeddings.word_embeddings = pruned_word_embeddings
        else:
            model.embeddings = pruned_word_embeddings




class XLMStructure(ModelStructure):
    MODEL_PREFIX: str = "transformer."
    ENCODER_PREFIX: str = r"attention.[0-9]+\."
    LAYER_PATTERNS = dict(
        query=r"attentions\.[0-9]+\.q_lin",
        key=r"attentions\.[0-9]+\.k_lin",
        value=r"attentions\.[0-9]+\.v_lin",
        att_dense=r"attentions\.[0-9]+\.out_lin",
        interm_dense=r"ffns\.[0-9]+\.lin1",
        output_dense=r"ffns\.[0-9]+\.lin2",
    )
    ATTENTION_PREFIX = (r"attentions\.[0-9]",)
    ATTENTION_LAYERS = ("q_lin", "k_lin", "v_lin")
    MHA_LAYERS = ATTENTION_LAYERS + ("att_dense",)
    NAME_CONFIG = dict(
        hidden_size="emb_dim",
        intermediate_size="emb_dim",
        num_hidden_layers="n_layers",
        num_attention_heads="n_heads",
        attention_head_size="attention_head_size",
    )