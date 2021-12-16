import logging
logger = logging.getLogger(__name__)

from torch.utils.data import SequentialSampler, DataLoader
from transformers import XLMRobertaTokenizer

import os
import torch
from torch.utils.data import TensorDataset, Dataset, ConcatDataset
from typing import List
import csv, json
from io import open
from dataclasses import dataclass
import dataclasses
from typing import List,Optional,Union


class MultilingualNLIDataset(Dataset):
    def __init__(self, task: str, data_dir: str, split: str, prefix: str, max_seq_length: int, langs: List[str], local_rank=-1, tokenizer=None, reuse_cache=False):
        print("Init NLIDataset")
        self.split = split
        self.processor = processors[task]()
        self.output_mode = output_modes[task]
        self.cached_features_files = {lang : os.path.join(data_dir, f'{prefix}_{split}_{max_seq_length}_{lang}.tensor') for lang in langs}
        self.lang_datasets = {}


        for lang, cached_features_file in self.cached_features_files.items():
            if os.path.exists(cached_features_file) and reuse_cache is True:
                logger.info("Loading features from cached file %s", cached_features_file)
                features_tensor = torch.load(cached_features_file)
            else:
                logger.info("Creating features from dataset file at %s", cached_features_file)
                label_list = self.processor.get_labels()
                examples = self.processor.get_examples(data_dir, split,lang=lang)
                features_tensor = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, self.output_mode)
                if local_rank in [-1, 0]:
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(features_tensor, cached_features_file)
            features_tensor = features_tensor[:-1] + (features_tensor[-1].long(),)
            self.lang_datasets[lang] = TensorDataset(*features_tensor)
        self.all_dataset = ConcatDataset(list(self.lang_datasets.values()))

    def __getitem__(self, index):
        example = self.all_dataset[index]
        input_ids, attention_mask, token_type_ids, labels = example
        return {'input_ids':input_ids, 'attention_mask':attention_mask, 
                'token_type_ids':token_type_ids, 'labels':labels}

    def __len__(self):
        return len(self.all_dataset)

class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_examples(self, data_dir, split, **kwargs):
        if split=='train':
            return self.get_train_examples(data_dir=data_dir, **kwargs)
        elif split=='dev':
            return self.get_dev_examples(data_dir=data_dir, **kwargs)
        elif split=='test':
            return self.get_test_examples(data_dir=data_dir, **kwargs)
        else:
            raise ValueError

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class XnliProcessor(DataProcessor):

    def get_dev_examples(self, lang, data_dir):
        examples = []
        input_file = os.path.join(data_dir,'xnli.dev.jsonl')
        with open(input_file,'r',encoding='utf-8-sig') as f:
           for index,line in enumerate(f):
               raw_example = json.loads(line)
               if raw_example['language'] != lang:
                   continue
               else:
                   text_a = raw_example['sentence1']
                   text_b = raw_example['sentence2']
                   label  = raw_example['gold_label']
                   guid   = f"dev-{index}"
                   examples.append(
                       InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def get_train_examples(self, lang, data_dir):
        examples = []
        input_file = os.path.join(data_dir,f'multinli.train.{lang}.tsv')
        with open(input_file,'r',encoding='utf-8-sig') as f:
            for index,line in enumerate(f):
                if index == 0:
                    continue
                line = line.strip().split('\t')
                guid = f"train-{index}"
                text_a = line[0]
                text_b = line[1]
                label = line[2]
                if label=='contradictory':
                    label = 'contradiction'
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, lang, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "xnli.test.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            language = line[0]
            if language != lang:
                continue
            guid = "%s-%s" % ("test", i)
            text_a = line[6]
            text_b = line[7]
            label = line[1]
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class PawsxProcessor(DataProcessor):

    def get_dev_examples(self, lang, data_dir):
        examples = []
        input_file = os.path.join(data_dir,f'dev-{lang}.tsv')
        with open(input_file,'r',encoding='utf-8-sig') as f:
           for index,line in enumerate(f):  
                text_a,text_b, label = line.strip().split('\t')
                guid   = f"dev-{index}"
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self,lang,data_dir):
        examples = []
        input_file = os.path.join(data_dir,f'test-{lang}.tsv')
        with open(input_file,'r',encoding='utf-8-sig') as f:
           for index,line in enumerate(f):  
                text_a,text_b, label = line.strip().split('\t')
                guid   = f"test-{index}"
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return ["0","1"]

    def get_train_examples(self, lang, data_dir):
        examples = []
        input_file = os.path.join(data_dir,f'translate-train/{lang}.tsv')
        with open(input_file,'r',encoding='utf-8-sig') as f:
            for index,line in enumerate(f):
                line = line.strip().split('\t')
                if len(line)==5:
                    text_a = line[2]
                    text_b = line[3]
                    label = line[4]
                elif len(line)==3:
                    text_a = line[0]
                    text_b = line[1]
                    label = line[2]
                else:
                    raise ValueError
                guid = f"train-{index}"
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



@dataclass(frozen=True)
class InputFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None

    def to_json_string(self):
        return json.dumps(dataclasses.asdict(self)) + "\n"


def convert_examples_to_features(examples, label_list, max_length,
                                 tokenizer, output_mode):
    if max_length is None:
        max_length = tokenizer.max_len

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample):
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_token_type_ids=True,
        return_tensors='pt'
    )
    label_ids = torch.tensor(labels, dtype=torch.long)

    features_tensors = (batch_encoding['input_ids'], batch_encoding['attention_mask'],
                        batch_encoding['token_type_ids'], label_ids)
    return features_tensors

processors = {
    "xnli": XnliProcessor,
    "pawsx": PawsxProcessor
}

output_modes = {
    "xnli": "classification",
    "pawsx":"classification",
}