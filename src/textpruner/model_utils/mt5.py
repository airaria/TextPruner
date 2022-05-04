import torch
from torch import nn

from .model_structure import ModelStructure
from .utils import DefaultModelVocabResizer


class MT5VocabResizer(DefaultModelVocabResizer):
    model_name : str = 'mt5'

    @classmethod
    def set_embeddings(cls, model, token_ids):
        def _prun(old_weight, token_ids):
            pruned_word_embeddings_weight = torch.index_select(
                old_weight, 0, index=torch.LongTensor(token_ids).to(old_weight.device))
            return pruned_word_embeddings_weight 

        vocab_size = model.shared.weight.shape[0]
        max_token_ids = token_ids[-1]
        tokens_in_embed_notin_tokenizer_ids = list(range(max_token_ids+1, vocab_size))
        token_ids_temp = token_ids[:]
        token_ids_temp.extend(tokens_in_embed_notin_tokenizer_ids)

        model.config.vocab_size = len(token_ids_temp)

        old_word_embeddings_shared = model.shared

        old_word_embeddings_shared_weight = old_word_embeddings_shared.weight
        pruned_word_embeddings_shared_weight = _prun(old_word_embeddings_shared_weight, token_ids_temp)

        pruned_num_tokens, embedding_dim = pruned_word_embeddings_shared_weight.shape

        pruned_word_embeddings_shared = nn.Embedding(
            pruned_num_tokens, embedding_dim).to(old_word_embeddings_shared_weight.device)
        pruned_word_embeddings_shared.weight.data[:] = pruned_word_embeddings_shared_weight[:]
        
        model.shared = pruned_word_embeddings_shared
        model.encoder.embed_tokens = model.shared
        model.decoder.embed_tokens = model.shared

class MT5Structure(ModelStructure):
    MODEL_PREFIX: str = "transformer."
    ENCODER_PREFIX: str = r"encoder.block.[0-9]+\.layer."
    LAYER_PATTERNS = dict(
        query="0.SelfAttention.q",
        key="0.SelfAttention.k",
        value="0.SelfAttention.v",
        att_dense="0.SelfAttention.o",
        interm_dense="1.DenseReluDense.wi",
        output_dense="1.DenseReluDense.wo",
    )
    ATTENTION_PREFIX = ("0.SelfAttention",)
    ATTENTION_LAYERS = ("q", "k", "v")
    MHA_LAYERS = ATTENTION_LAYERS + ("att_dense",)
    NAME_CONFIG = dict(
        hidden_size="d_model",
        intermediate_size="d_ff",
        num_hidden_layers="num_layers",
        num_attention_heads="num_heads",
        attention_head_size="",
    )
