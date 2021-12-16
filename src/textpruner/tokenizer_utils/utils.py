from itertools import chain
from collections import Counter
from collections.abc import Iterable
from typing import Callable, Optional
from tqdm import tqdm
import logging
import json
logger = logging.getLogger(__name__)

def count_frequency(self, texts : Iterable):
    token_counter = Counter()

    for text in texts:
        tokens = self.tokenizer.tokenize(text)
        token_counter.update(tokens)
    all_tokens = [k for (k, v) in token_counter.most_common()]
    all_token_indices = self.tokenizer.convert_tokens_to_ids(all_tokens)
    return all_tokens, all_token_indices



def count_unique_tokens(dataiter, tokenizer, fn : Optional[Callable] =None) -> Counter :
    assert not isinstance(dataiter,str), "dataiter is assumed to be a collection (list, tuple, ...) of strings, not a single string"
    token_ids = Counter()
    for item in tqdm(dataiter):
        if fn is not None:
            item = fn(item) # pre-transform
        if isinstance(item, str):
            token_ids.update(tokenizer.encode(item, add_special_tokens=True))
        else:
            assert isinstance(item[0],str) # list of string
            token_ids.update(list(chain(*(tokenizer.encode(i, add_special_tokens=True) for i in item))))
    return token_ids