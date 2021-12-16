# Pruning the Classification model

These scripts perform transformer pruning on the classification model (`XLMRobertaForSequenceClassification`) and evaluate the performance.

Download the fine-tuned model or train your own model on PAWS-X dataset, and save the files to `../models/xlmr_pawsx`.

Download link: 
    * [Google Drive](https://drive.google.com/drive/folders/1TXuIvcYJ0aje7WC-LyrxstzeJn4_383r?usp=sharing)
    * [Hugging Face Models](https://huggingface.co/ziqingyang/XLMRobertaBaseForPAWSX-en/tree/main)

* Pruning with the textpruner-CLI tool:
```bash
bash transformer_pruning.sh
```

* Pruning with the python script:
```bash
MODEL_PATH=../models/xlmr_pawsx
python transformer_pruning.py $MODEL_PATH 
```

* Evaluate the model:

Set `$PRUNED_MODEL_PATH` to the directory where the pruned model is stored.

```bash
cp $MODEL_PATH/sentencepiece.bpe.model $PRUNED_MODEL_PATH
python measure_performance.py $PRUNED_MODEL_PATH
```


# Pruning the Classification model with masks

This scripts perform transformer pruning on the classification model with the given (random) masks

```bash
MODEL_PATH=../models/xlmr_pawsx
python transformer_pruning_with_masks.py $MODEL_PATH 
```