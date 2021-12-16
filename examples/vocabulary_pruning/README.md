# Pruning the Classification model

These scripts perform vocabulary pruning on the classification model (`XLMRobertaForSequenceClassification`) and evaluate the performance.

We use a subset of XNLI English training set as the vocabulary file.

Download the fine-tuned model or train your own model on PAWS-X dataset, and save the files to `../models/xlmr_pawsx`.

Download link: 
    * [Google Drive](https://drive.google.com/drive/folders/1TXuIvcYJ0aje7WC-LyrxstzeJn4_383r?usp=sharing)
    * [Hugging Face Models](https://huggingface.co/ziqingyang/XLMRobertaBaseForPAWSX-en/tree/main)

* Pruning with the textpruner-CLI tool:
```bash
bash vocabulary_pruning.sh
```

* Pruning with the python script:
```bash
VOCABULARY_FILE=../datasets/xnli/en.tsv
MODEL_PATH=../models/xlmr_pawsx
python vocabulary_pruning.py $MODEL_PATH $VOCABULARY_FILE
```

* Evaluate the model:

Set `$PRUNED_MODEL_PATH` to the directory where the pruned model is stored.

```bash
python measure_performance.py $PRUNED_MODEL_PATH
```

# Pruning the Pre-Trained models for MLM

This script prunes the pre-trained models for MLM with a vocabulary limited to the SST-2 training set.

Set `$MODEL_PATH` to the directory where the pre-trained model （BERT, RoBERTa, etc.) is stored.

```bash
python MaskedLM_vocabulary_pruning.py $MODEL_PATH
```


