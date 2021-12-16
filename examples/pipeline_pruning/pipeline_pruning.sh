textpruner-cli  \
  --pruning_mode pipeline \
  --configurations ../configurations/tc-iterative.json \
  --model_class XLMRobertaForSequenceClassification \
  --tokenizer_class XLMRobertaTokenizer \
  --model_path ../models/xlmr_pawsx \
  --vocabulary ../datasets/xnli/en.tsv \
  --dataloader_and_adaptor ../classification_utils/dataloader_script.py