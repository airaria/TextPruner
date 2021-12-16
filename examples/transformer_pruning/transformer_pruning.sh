textpruner-cli  \
  --pruning_mode transformer \
  --configurations ../configurations/tc-iterative.json \
  --model_class XLMRobertaForSequenceClassification \
  --tokenizer_class XLMRobertaTokenizer \
  --model_path ../models/xlmr_pawsx \
  --dataloader_and_adaptor ../classification_utils/dataloader_script.py

