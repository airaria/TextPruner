import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from transformers import XLMRobertaForSequenceClassification,XLMRobertaTokenizer
import sys, os

sys.path.insert(0, os.path.abspath('..'))

from classification_utils.my_dataset import MultilingualNLIDataset
from classification_utils.predict_function import predict

model_path = sys.argv[1]
taskname = 'xnli'
data_dir = '../datasets/xnli'
split = 'test'
max_seq_length=128
eval_langs = ['en']
batch_size=32
device = 'cuda'

# Re-initialze the tokenizer
model = XLMRobertaForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
eval_dataset = MultilingualNLIDataset(
    task=taskname, data_dir=data_dir, split=split, prefix='xlmr',
    max_seq_length=max_seq_length, langs=eval_langs,  tokenizer=tokenizer)
eval_datasets = [eval_dataset.lang_datasets[lang] for lang in eval_langs]
predict(model, eval_datasets, eval_langs, device, batch_size)
