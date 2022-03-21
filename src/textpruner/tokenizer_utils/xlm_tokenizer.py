import os
from .utils import count_unique_tokens
import logging
import json
logger = logging.getLogger(__name__)

class XLMTokenizer:
    @staticmethod
    def get_token_ids(tokenizer, dataiter=None, additional_tokens=None, additional_token_ids=None, min_count=1):
        token_ids = []
        # add special tokens
        special_tokens = ['<s>', '</s>', '<pad>', '<unk>', '<special1>', '<special0>', '<special2>', '<special3>', '<special4>', '<special5>', '<special6>', '<special7>', '<special8>', '<special9>']
        special_token_ids = list(range(0, 14))
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
        assert len(token_ids) == len(set(token_ids))

        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        token_dict = {}
        for i in range(len(tokens)):
            token_dict[tokens[i]] = i
        

        tokenizer.save_pretrained(outdir)
        pruned_vocab_file = os.path.join(outdir, 'vocab.json')
        with open(pruned_vocab_file, 'w', encoding='utf-8') as f:
            json.dump(token_dict, f)

        print(f"New embedding size {len(token_ids)} pruned vocab file has been saved to {pruned_vocab_file}. Reintialize the tokenizer!")

        
        bpe_ranks = sorted(tokenizer.bpe_ranks.items(), key = lambda k: k[1])

        
        pruned_merges_file = os.path.join(outdir, 'merges.txt')
        with open(pruned_merges_file, "w", encoding="utf-8") as writer:
            for bpe_tokens, _ in bpe_ranks:
                if len(bpe_tokens) != 2:
                    continue
                writer.write(bpe_tokens[0] + " " + bpe_tokens[1] + "\n")
        
        
            
