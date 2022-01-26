# Pruning the Classification model

These scripts perform vocabulary pruning on the classification model (`XLMRobertaForSequenceClassification`) and evaluate the performance.

We use the English and Chinese training sets as the vocabulary file.

Download the fine-tuned model or train your own model on XNLI dataset, and save the files to `../models/xlmr_xnli`.

Download link: 
    * [Hugging Face Models](https://huggingface.co/ziqingyang/XLMRobertaBaseForXNLI-en/tree/main)

See the README in ../datasets/xnli for how to construct the dataset.

* Pruning with the python script:
```bash
VOCABULARY_FILE=../datasets/xnli/multinli.train.en_zh.tsv
MODEL_PATH=../models/xlmr_xnli
python vocabulary_pruning.py $MODEL_PATH $VOCABULARY_FILE
```

* Evaluate the model:

Set `$PRUNED_MODEL_PATH` to the directory where the pruned model is stored.

```bash
python measure_performance.py $PRUNED_MODEL_PATH
```