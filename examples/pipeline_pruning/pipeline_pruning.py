import logging

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from transformers import XLMRobertaForSequenceClassification,XLMRobertaTokenizer
from textpruner import summary, PipelinePruner, TransformerPruningConfig
from textpruner.commands.utils import read_file_line_by_line
import sys, os

sys.path.insert(0, os.path.abspath('..'))
from classification_utils.dataloader_script import dataloader, MultilingualNLIDataset
from classification_utils.predict_function import predict

# Initialize your model and load data
model_path = sys.argv[1]
vocabulary = sys.argv[2]
model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
texts, _ = read_file_line_by_line(vocabulary)

print("Before pruning:")
print(summary(model))

transformer_pruning_config = TransformerPruningConfig(
    target_ffn_size=2048, target_num_of_heads=8, 
    pruning_method='iterative',n_iters=1)
pruner = PipelinePruner(model, tokenizer, transformer_pruning_config=transformer_pruning_config)
pruner.prune(dataloader=dataloader, dataiter=texts, save_model=True)

print("After pruning:")
print(summary(model))
for i in range(12):
    print ((model.base_model.encoder.layer[i].intermediate.dense.weight.shape,
            model.base_model.encoder.layer[i].intermediate.dense.bias.shape,
            model.base_model.encoder.layer[i].attention.self.key.weight.shape))


print("Measure performance")

taskname = 'pawsx'
data_dir = '../datasets/pawsx'
split = 'test'
max_seq_length=128
eval_langs = ['en']
batch_size=32
device= model.device

# Re-initialze the tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained(pruner.save_dir)
eval_dataset = MultilingualNLIDataset(
    task=taskname, data_dir=data_dir, split=split, prefix='xlmr',
    max_seq_length=max_seq_length, langs=eval_langs,  tokenizer=tokenizer)
eval_datasets = [eval_dataset.lang_datasets[lang] for lang in eval_langs]

predict(model, eval_datasets, eval_langs, device, batch_size)
