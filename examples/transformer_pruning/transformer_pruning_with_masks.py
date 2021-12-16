import textpruner
from transformers import XLMRobertaForSequenceClassification,XLMRobertaTokenizer
import datasets
from textpruner import summary, TransformerPruner
import sys

# Initialize your model and load your data
model_path = sys.argv[1]
model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)

print("Before pruning:")
print(summary(model))

pruner = TransformerPruner(model)

ffn_mask = textpruner.pruners.utils.random_mask_tensor((12,3072))
head_mask = textpruner.pruners.utils.random_mask_tensor((12,12), even_masks=False)

pruner.prune(head_mask=head_mask, ffn_mask=ffn_mask,save_model=True)

print("After pruning:")
print(summary(model))

for i in range(12):
    print ((model.base_model.encoder.layer[i].intermediate.dense.weight.shape,
            model.base_model.encoder.layer[i].intermediate.dense.bias.shape,
            model.base_model.encoder.layer[i].attention.self.key.weight.shape))