
import os
from .utils import count_unique_tokens
import logging
logger = logging.getLogger(__name__)
try:
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
except ImportError:
    logger.warning("Could not import sentencepiece. Pruning embeddings of sentencepiece-based model is not available.")


class XLMRSentencepieceTokenizer:

    @staticmethod
    def get_token_ids(tokenizer, dataiter=None, additional_tokens=None, additional_token_ids=None, min_count=1):
        token_ids = []
        # add special tokens
        # should equal to [0,1,2,3,size +1]
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
        token_ids = sorted(special_token_ids + normal_token_ids) # to make sure [0,1,2,3, ...., <mask>]
        return token_ids

    @staticmethod
    def save_vocab(tokenizer, token_ids, outdir):
        fairseq_offset = 1
        # {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
        fairseq_special_tokens_ids = [0, 1, 2, 3]
        fairseq_special_tokens_ids.append(
            len(tokenizer.sp_model) + fairseq_offset)  # ["<mask>"]
        # remove special tokens
        token_ids = [
            t for t in token_ids if t not in fairseq_special_tokens_ids]

        # special tokens + normal tokens
        spm_token_ids = [0, 1, 2] + \
            [t-fairseq_offset for t in token_ids]
        assert len(spm_token_ids) == len(set(spm_token_ids))


        m = sp_pb2_model.ModelProto()
        m.ParseFromString(tokenizer.sp_model.serialized_model_proto())

        spm_tokens = set([m.pieces[i].piece for i in spm_token_ids])
        new_pieces = [p for p in m.pieces if p.piece in spm_tokens]

        # delete all
        del m.pieces[:]
        m.pieces.extend(new_pieces)

        # #debug
        # #debug
        # print ("spm_token_ids:",spm_token_ids)
        # print ("spm_tokens:",spm_tokens)
        # print ('new pieces:',[p.piece for p in m.pieces])

        pruned_vocab_file = os.path.join(outdir, 'sentencepiece.bpe.model')
        with open(pruned_vocab_file, 'wb') as f:
            f.write(m.SerializeToString())
        print(f"New embedding size {len(new_pieces)+2} pruned vocab file has been saved to {pruned_vocab_file}. Reintialize the tokenizer!")
