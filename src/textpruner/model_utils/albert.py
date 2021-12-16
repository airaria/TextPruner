from .utils import DefaultModelVocabResizer
from .model_structure import ModelStructure

class AlbertVocabResizer(DefaultModelVocabResizer):
    model_name : str = 'albert'

class AlbertStructure(ModelStructure):
    MODEL_PREFIX: str = "albert."
    ENCODER_PREFIX: str = r"encoder.albert_layer_groups.0.albert_layers.0."
    LAYER_PATTERNS = dict(
        query="attention.query",
        key="attention.key",
        value="attention.value",
        att_dense="attention.dense",
        interm_dense=r"ffn$",
        output_dense="ffn_output",
    )
    ATTENTION_PREFIX = ("attention",)
    ATTENTION_LAYERS = ("query", "key", "value")
    MHA_LAYERS = ATTENTION_LAYERS + ("att_dense",)
    NAME_CONFIG = dict(
        hidden_size="hidden_size",
        intermediate_size="intermediate_size",
        num_hidden_layers="num_hidden_layers",
        num_attention_heads="num_attention_heads",
        attention_head_size="attention_head_size",
    )