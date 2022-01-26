import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from transformers import XLMRobertaForSequenceClassification,XLMRobertaTokenizer
from textpruner import summary, TransformerPruner, TransformerPruningConfig
import sys, os

sys.path.insert(0, os.path.abspath('..'))

from classification_utils.dataloader_script_xnli import dataloader, eval_langs, batch_size,MultilingualNLIDataset
from classification_utils.predict_function import predict

model_path = sys.argv[1]
model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)

print("Before pruning:")
print(summary(model))

def adatpor(model_outputs):
    logits = model_outputs.logits
    return logits #entropy(logits)


transformer_pruning_config = TransformerPruningConfig(
    target_ffn_size=1536, target_num_of_heads=6, 
    pruning_method='iterative',n_iters=8,use_logits=True,head_even_masking=False,ffn_even_masking=False)
pruner = TransformerPruner(model,transformer_pruning_config=transformer_pruning_config)   
pruner.prune(dataloader=dataloader, save_model=False, adaptor=adatpor)

# save the tokenizer to the same place
#tokenizer.save_pretrained(pruner.save_dir)

print("After pruning:")
print(summary(model))

for i in range(12):
    print ((model.base_model.encoder.layer[i].intermediate.dense.weight.shape,
            model.base_model.encoder.layer[i].intermediate.dense.bias.shape,
            model.base_model.encoder.layer[i].attention.self.key.weight.shape))


print("Measure performance")
taskname = 'xnli'
data_dir = '../datasets/xnli'
split = 'dev'
max_seq_length=128
eval_langs = ['en','zh']
batch_size=32
device= model.device
eval_dataset = MultilingualNLIDataset(
    task=taskname, data_dir=data_dir, split=split, prefix='xlmr',
    max_seq_length=max_seq_length, langs=eval_langs,  tokenizer=tokenizer)
eval_datasets = [eval_dataset.lang_datasets[lang] for lang in eval_langs]
print("dev")
predict(model, eval_datasets, eval_langs, device, batch_size)

split="test"
print("test")
eval_dataset = MultilingualNLIDataset(
    task=taskname, data_dir=data_dir, split=split, prefix='xlmr',
    max_seq_length=max_seq_length, langs=eval_langs,  tokenizer=tokenizer)
eval_datasets = [eval_dataset.lang_datasets[lang] for lang in eval_langs]
predict(model, eval_datasets, eval_langs, device, batch_size)

print(transformer_pruning_config)