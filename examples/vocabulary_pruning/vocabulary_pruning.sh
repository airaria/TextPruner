textpruner-cli  \
  --pruning_mode vocabulary \
  --configurations ../configurations/vc.json ../configurations/gc.json \
  --model_class XLMRobertaForSequenceClassification \
  --tokenizer_class XLMRobertaTokenizer \
  --model_path ../models/xlmr_pawsx \
  --vocabulary ../datasets/xnli/en.tsv