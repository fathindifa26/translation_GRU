# src/data/preprocessing.py
from transformers import BertTokenizer
from torchtext.vocab import build_vocab_from_iterator

def tokenize_gru(example, tokenizer, max_length=1000, lower=True, sos_token="<sos>", eos_token="<eos>"):
    en_text = example["indonesia"].lower() if lower else example["indonesia"]
    id_text = example["jawa"].lower() if lower else example["jawa"]
    en_tokens = [sos_token] + tokenizer.tokenize(en_text)[:max_length] + [eos_token]
    id_tokens = [sos_token] + tokenizer.tokenize(id_text)[:max_length] + [eos_token]
    return {"en_tokens": en_tokens, "id_tokens": id_tokens}

def numericalize_example(example, en_vocab, id_vocab):
    en_ids = en_vocab.lookup_indices(example["en_tokens"])
    id_ids = id_vocab.lookup_indices(example["id_tokens"])
    return {"en_ids": en_ids, "id_ids": id_ids}

def build_vocabulary(dataset, min_freq=1, special_tokens=["<unk>", "<pad>", "<sos>", "<eos>"]):
    en_vocab = build_vocab_from_iterator(
        [example["en_tokens"] for example in dataset["train"]],
        min_freq=min_freq,
        specials=special_tokens,
    )
    id_vocab = build_vocab_from_iterator(
        [example["id_tokens"] for example in dataset["train"]],
        min_freq=min_freq,
        specials=special_tokens,
    )
    en_vocab.set_default_index(en_vocab["<unk>"])
    id_vocab.set_default_index(id_vocab["<unk>"])
    return en_vocab, id_vocab

