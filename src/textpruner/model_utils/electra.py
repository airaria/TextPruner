from .utils import DefaultModelVocabResizer
from .model_structure import ModelStructure

class ElectraVocabResizer(DefaultModelVocabResizer):
    model_name : str = 'electra'

class ElectraStructure(ModelStructure):
    MODEL_PREFIX: str = "electra."
    ENCODER_PREFIX: str = r"encoder.layer.[0-9]+\."
    LAYER_PATTERNS = dict(
        query="attention.self.query",
        key="attention.self.key",
        value="attention.self.value",
        att_dense="attention.output.dense",
        interm_dense="intermediate.dense",
        output_dense="output.dense",
    )
    ATTENTION_PREFIX = ("attention.self",)
    ATTENTION_LAYERS = ("query", "key", "value")
    MHA_LAYERS = ATTENTION_LAYERS + ("att_dense",)
    NAME_CONFIG = dict(
        hidden_size="hidden_size",
        intermediate_size="intermediate_size",
        num_hidden_layers="num_hidden_layers",
        num_attention_heads="num_attention_heads",
        attention_head_size="attention_head_size",
    )