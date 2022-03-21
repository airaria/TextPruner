from .utils import DefaultModelVocabResizer
from .model_structure import ModelStructure
import torch 
from torch import nn 
class BartVocabResizer(DefaultModelVocabResizer):
    model_name : str = 'bart'

    @classmethod
    def set_embeddings(cls, model, token_ids):
        def _prun(old_weight, token_ids):
            pruned_word_embeddings_weight = torch.index_select(
                old_weight, 0, index=torch.LongTensor(token_ids).to(old_weight.device))     
            return pruned_word_embeddings_weight 

        old_word_embeddings_shared, old_word_embeddings_encoder, old_word_embeddings_decoder = \
            model.shared, model.encoder.embed_tokens, model.decoder.embed_tokens

        old_word_embeddings_shared_weight, old_word_embeddings_encoder_weight, old_word_embeddings_decoder_weight = \
            old_word_embeddings_shared.weight, old_word_embeddings_encoder.weight, old_word_embeddings_decoder.weight

        pruned_word_embeddings_shared_weight, pruned_word_embeddings_encoder_weight, pruned_word_embeddings_decoder_weight = \
            _prun(old_word_embeddings_shared_weight, token_ids), _prun(old_word_embeddings_encoder_weight, token_ids), _prun(old_word_embeddings_decoder_weight, token_ids)

        pruned_num_tokens, embedding_dim = pruned_word_embeddings_shared_weight.shape

        pruned_word_embeddings_shared = nn.Embedding(
            pruned_num_tokens, embedding_dim).to(old_word_embeddings_shared_weight.device)
        pruned_word_embeddings_shared.weight.data[:] = pruned_word_embeddings_shared_weight[:]

        pruned_word_embeddings_encoder = nn.Embedding(
            pruned_num_tokens, embedding_dim).to(old_word_embeddings_shared_weight.device)
        pruned_word_embeddings_encoder.weight.data[:] = pruned_word_embeddings_encoder_weight[:]

        pruned_word_embeddings_decoder = nn.Embedding(
            pruned_num_tokens, embedding_dim).to(old_word_embeddings_shared_weight.device)
        pruned_word_embeddings_decoder.weight.data[:] = pruned_word_embeddings_decoder_weight[:]
        
        model.shared = pruned_word_embeddings_shared
        model.encoder.embed_tokens = pruned_word_embeddings_encoder
        model.decoder.embed_tokens = pruned_word_embeddings_decoder    

class BartStructure(ModelStructure):
    MODEL_PREFIX: str = "model."
    ENCODER_PREFIX: str = r"encoder.layers.[0-9]+\."
    LAYER_PATTERNS = dict(
        query="self_attn.q_proj",
        key="self_attn.k_proj",
        value="self_attn.v_proj",
        att_dense="self_attn.out_proj",
        interm_dense="fc1",
        output_dense="fc2",
    )
    ATTENTION_PREFIX = ("self_attn",)
    ATTENTION_LAYERS = ("q_proj", "k_proj", "v_proj")
    MHA_LAYERS = ATTENTION_LAYERS + ("att_dense",)
    NAME_CONFIG = dict(
        hidden_size="d_model",
        intermediate_size="encoder_ffn_dim",
        num_hidden_layers="encoder_layers",
        num_attention_heads="num_attention_heads",
        attention_head_size="",
    )