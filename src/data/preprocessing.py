import torch
import torchtext
from transformers import BertTokenizer
from datasets import Dataset

def preprocess_dataset(dataset, max_length=1000, lower=True, min_freq=1):
    """
    Preprocessing dataset untuk diterapkan pada model.
    
    Args:
        dataset (DatasetDict): Dataset yang berisi 'train', 'validation', dan 'test'.
        max_length (int): Panjang maksimum tokenisasi.
        lower (bool): Apakah teks akan dikonversi ke huruf kecil.
        min_freq (int): Minimum frekuensi token untuk dimasukkan ke vocabulary.

    Returns:
        train_data, valid_data, test_data: Dataset yang sudah diproses.
        en_vocab, id_vocab: Vocabulary dari dataset.
    """

    # Load tokenizer dari Hugging Face
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    # Token khusus
    unk_token = "<unk>"
    pad_token = "<pad>"
    sos_token = "<sos>"
    eos_token = "<eos>"
    special_tokens = [unk_token, pad_token, sos_token, eos_token]

    def tokenize_example(example):
        en_text = example["text_1"]  # Bahasa Jawa
        id_text = example["text_2"]  # Bahasa Indonesia

        if lower:
            en_text = en_text.lower()
            id_text = id_text.lower()

        # Tokenisasi menggunakan BERT tokenizer (subword tokenization)
        en_tokens = [sos_token] + tokenizer.tokenize(en_text)[:max_length] + [eos_token]
        id_tokens = [sos_token] + tokenizer.tokenize(id_text)[:max_length] + [eos_token]

        # Jika token kosong, isi dengan <unk>
        if len(en_tokens) <= 2:  # <sos> dan <eos> saja
            en_tokens = [sos_token, "<unk>", eos_token]
        if len(id_tokens) <= 2:
            id_tokens = [sos_token, "<unk>", eos_token]

        return {"en_tokens": en_tokens, "id_tokens": id_tokens}

    # Tokenisasi dataset
    train_data = dataset["train"].map(tokenize_example)
    valid_data = dataset["validation"].map(tokenize_example)
    test_data = dataset["test"].map(tokenize_example)

    print("✅ Tokenisasi selesai dengan BERT tokenizer!")

    # Membangun vocabulary dari token yang telah dibuat
    en_vocab = torchtext.vocab.build_vocab_from_iterator(
        (example["en_tokens"] for example in train_data),
        min_freq=min_freq,
        specials=special_tokens,
    )
    id_vocab = torchtext.vocab.build_vocab_from_iterator(
        (example["id_tokens"] for example in train_data),
        min_freq=min_freq,
        specials=special_tokens,
    )

    # Set default index untuk <unk>
    unk_index = en_vocab[unk_token]
    en_vocab.set_default_index(unk_index)
    id_vocab.set_default_index(unk_index)

    def numericalize_example(example):
        en_ids = en_vocab.lookup_indices(example["en_tokens"])
        id_ids = id_vocab.lookup_indices(example["id_tokens"])
        return {"en_ids": en_ids, "id_ids": id_ids}

    # Konversi token menjadi indeks numerik
    train_data = train_data.map(numericalize_example)
    valid_data = valid_data.map(numericalize_example)
    test_data = test_data.map(numericalize_example)

    # Set format untuk PyTorch
    format_columns = ["en_ids", "id_ids"]
    train_data = train_data.with_format("torch", columns=format_columns, output_all_columns=True)
    valid_data = valid_data.with_format("torch", columns=format_columns, output_all_columns=True)
    test_data = test_data.with_format("torch", columns=format_columns, output_all_columns=True)

    print("✅ Data siap digunakan dalam format PyTorch!")

    return train_data, valid_data, test_data, en_vocab, id_vocab
