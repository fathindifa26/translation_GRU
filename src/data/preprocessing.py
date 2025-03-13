import torch  # Library utama PyTorch untuk deep learning
import torchtext  # Library untuk pemrosesan teks di PyTorch
import re  # Library untuk ekspresi reguler dalam pembersihan teks
from datasets import Dataset  # Library untuk mengelola dataset

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

    # Definisi token khusus
    unk_token = "<unk>"  # Token untuk kata yang tidak dikenal
    pad_token = "<pad>"  # Token untuk padding
    sos_token = "<sos>"  # Token untuk awal kalimat
    eos_token = "<eos>"  # Token untuk akhir kalimat
    special_tokens = [unk_token, pad_token, sos_token, eos_token]

    def clean_text(text):
        """
        Membersihkan teks dengan menghapus karakter yang tidak diinginkan.
        """
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Menghapus karakter selain huruf dan angka
        text = re.sub(r"\s+", " ", text).strip()  # Menghapus spasi berlebih
        return text

    def tokenize_example(example):
        """
        Fungsi untuk melakukan pembersihan dan tokenisasi sederhana (perkataan) pada contoh dataset.
        """
        en_text = example["text_1"]  # Bahasa Jawa
        id_text = example["text_2"]  # Bahasa Indonesia

        if lower:
            en_text = en_text.lower()  # Konversi ke huruf kecil jika diperlukan
            id_text = id_text.lower()

        # Membersihkan teks dari karakter yang tidak diinginkan
        en_text = clean_text(en_text)
        id_text = clean_text(id_text)

        # Tokenisasi sederhana (berdasarkan kata, bukan subword)
        en_tokens = [sos_token] + en_text.split()[:max_length] + [eos_token]
        id_tokens = [sos_token] + id_text.split()[:max_length] + [eos_token]

        # Jika token kosong, isi dengan <unk>
        if len(en_tokens) <= 2:  # Jika hanya terdiri dari <sos> dan <eos>
            en_tokens = [sos_token, "<unk>", eos_token]
        if len(id_tokens) <= 2:
            id_tokens = [sos_token, "<unk>", eos_token]

        return {"en_tokens": en_tokens, "id_tokens": id_tokens}

    # Tokenisasi dataset
    train_data = dataset["train"].map(tokenize_example)
    valid_data = dataset["validation"].map(tokenize_example)
    test_data = dataset["test"].map(tokenize_example)

    print("✅ Tokenisasi sederhana selesai!")

    # Membangun vocabulary dari token yang telah dibuat
    en_vocab = torchtext.vocab.build_vocab_from_iterator(
        (example["en_tokens"] for example in train_data),  # Menggunakan token Bahasa Jawa
        min_freq=min_freq,  # Token harus muncul minimal sebanyak `min_freq`
        specials=special_tokens,  # Menambahkan token khusus ke dalam vocab
    )
    id_vocab = torchtext.vocab.build_vocab_from_iterator(
        (example["id_tokens"] for example in train_data),  # Menggunakan token Bahasa Indonesia
        min_freq=min_freq,
        specials=special_tokens,
    )

    # Set default index untuk token yang tidak dikenal (<unk>)
    unk_index = en_vocab[unk_token]
    en_vocab.set_default_index(unk_index)
    id_vocab.set_default_index(unk_index)

    def numericalize_example(example):
        """
        Konversi token menjadi indeks numerik sesuai vocabulary.
        """
        en_ids = en_vocab.lookup_indices(example["en_tokens"])  # Konversi token Bahasa Jawa ke indeks
        id_ids = id_vocab.lookup_indices(example["id_tokens"])  # Konversi token Bahasa Indonesia ke indeks
        return {"en_ids": en_ids, "id_ids": id_ids}

    # Konversi token menjadi indeks numerik untuk seluruh dataset
    train_data = train_data.map(numericalize_example)
    valid_data = valid_data.map(numericalize_example)
    test_data = test_data.map(numericalize_example)

    # Set format dataset agar kompatibel dengan PyTorch
    format_columns = ["en_ids", "id_ids"]  # Kolom yang akan disimpan dalam format tensor
    train_data = train_data.with_format("torch", columns=format_columns, output_all_columns=True)
    valid_data = valid_data.with_format("torch", columns=format_columns, output_all_columns=True)
    test_data = test_data.with_format("torch", columns=format_columns, output_all_columns=True)

    print("✅ Data siap digunakan dalam format PyTorch!")

    return train_data, valid_data, test_data, en_vocab, id_vocab
