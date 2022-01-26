Download [XNLI](https://github.com/facebookresearch/XNLI) and put `multinli.train.en.tsv`,`multinli.train.zh.tsv`,`xnli.dev.tsv`,`xnli.test.tsv` here.

Concatenate the train files:
```bash
cat multinli.train.en.tsv multinli.train.zh.tsv > multinli.train.en_zh.tsv
```
