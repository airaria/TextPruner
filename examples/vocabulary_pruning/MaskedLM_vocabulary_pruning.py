from transformers import AutoModelForMaskedLM, AutoTokenizer
import datasets
import torch
from textpruner import summary, VocabularyPruner
import sys

# Initialize your model and load your data
model_path = sys.argv[1]
model = AutoModelForMaskedLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
texts = datasets.load_dataset('glue','sst2')['train']['sentence']

print("Before pruning:")
print(summary(model))

pruner = VocabularyPruner(model, tokenizer)
save_dir = pruner.prune(dataiter=texts, save_model=True)


print("After pruning:")
print(summary(model))



#  Check consistency

print ("Testing consistency")

samples = texts[:10]

old_model = AutoModelForMaskedLM.from_pretrained(model_path,output_hidden_states=True).eval()
new_model = AutoModelForMaskedLM.from_pretrained(save_dir,output_hidden_states=True).eval()

old_tokenizer=AutoTokenizer.from_pretrained(model_path, use_fast=False)
new_tokenizer=AutoTokenizer.from_pretrained(save_dir, use_fast=False)

old_inputs = old_tokenizer(samples,padding=True,return_tensors='pt')
new_inputs = new_tokenizer(samples,padding=True,return_tensors='pt')

with torch.no_grad():
    old_outputs = old_model(**old_inputs)
    new_outputs = new_model(**new_inputs)

max_deviation = torch.abs(old_outputs.hidden_states[-1] - new_outputs.hidden_states[-1]).max()
print (f"Max deviation between two models: {max_deviation}")