import os
from .utils import count_unique_tokens

class SubwordTokenizer:
    @staticmethod
    def get_token_ids(tokenizer, dataiter=None, additional_tokens=None, additional_token_ids=None, min_count=1):
        token_ids = []
        # add special tokens
        special_token_ids = list(tokenizer.all_special_ids)

        normal_token_ids = []
        if dataiter is not None:
            token_ids_counter = count_unique_tokens(dataiter, tokenizer)
            normal_token_ids += [k for k,v in token_ids_counter.items() if v >= min_count]
        if additional_tokens is not None and len(additional_tokens) > 0:
            normal_token_ids += list(
                tokenizer.convert_tokens_to_ids(additional_tokens))
        if additional_token_ids is not None and len(additional_token_ids) > 0:
            normal_token_ids += list(additional_token_ids)
        normal_token_ids = list(set(normal_token_ids)-set(special_token_ids))
        token_ids = sorted(special_token_ids + normal_token_ids)
        return token_ids

    @staticmethod
    def save_vocab(tokenizer, token_ids, outdir):
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        pruned_vocab_file = os.path.join(outdir, 'vocab.txt')
        with open(pruned_vocab_file, 'w', encoding='utf-8') as f:
            for token in tokens:
                f.write(token+'\n')
        print(f"New embedding size {len(token_ids)} pruned vocab file has been saved to {pruned_vocab_file}. Reintialize the tokenizer!")