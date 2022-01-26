# Pruning the Classification model

These scripts perform transformer pruning **in a self-supervised way** on the classification model (`XLMRobertaForSequenceClassification`) and evaluate the performance.

Download the fine-tuned model or train your own model on XNLI dataset, and save the files to `../models/xlmr_xnli`.

Download link: 
    * [Hugging Face Models](https://huggingface.co/ziqingyang/XLMRobertaBaseForXNLI-en/tree/main)

See the README in ../datasets/xnli for how to construct the dataset.

* Pruning with the python script:
```bash
MODEL_PATH=../models/xlmr_xnli
python transformer_pruning_selfsupervised.py $MODEL_PATH
```

* Evaluate the model:

Set `$PRUNED_MODEL_PATH` to the directory where the pruned model is stored.

```bash
cp $MODEL_PATH/sentencepiece.bpe.model $PRUNED_MODEL_PATH
python measure_performance.py $PRUNED_MODEL_PATH
```