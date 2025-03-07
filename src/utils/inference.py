# src/utils/inference.py

import torch

def translate_sentence(
    sentence,
    model,
    en_vocab,
    id_vocab,
    sos_token="<sos>",
    eos_token="<eos>",
    device="cpu",
    max_output_length=25,
    max_repetition=3,  # Batas maksimum pengulangan token tidak bermakna
):
    """
    Menerjemahkan kalimat dari bahasa Indonesia ke Jawa menggunakan model seq2seq.

    Args:
        sentence (str): Kalimat input dalam bahasa Indonesia.
        model (torch.nn.Module): Model terlatih.
        en_vocab (torchtext.vocab.Vocab): Vocabulary untuk bahasa target (Jawa).
        id_vocab (torchtext.vocab.Vocab): Vocabulary untuk bahasa sumber (Indonesia).
        sos_token (str): Token awal kalimat (default: "<sos>").
        eos_token (str): Token akhir kalimat (default: "<eos>").
        device (str): Perangkat yang digunakan (CPU atau GPU).
        max_output_length (int): Panjang maksimum keluaran.
        max_repetition (int): Batas pengulangan token tidak bermakna.

    Returns:
        str: Hasil terjemahan dalam bahasa Jawa.
    """

    model.eval()

    # Tokenisasi manual tanpa tokenizer (menggunakan vocabulary yang sudah dibuat)
    tokens = [sos_token] + sentence.lower().split()[:1000] + [eos_token]
    numericalized = [id_vocab[token] if token in id_vocab else id_vocab["<unk>"] for token in tokens]
    sentence_tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device)

    # Encoder
    with torch.no_grad():
        context = model.encoder(sentence_tensor)

    # Decoder
    trg_tokens = [en_vocab[sos_token]]
    hidden = context
    last_token = None
    repeat_count = 0

    for _ in range(max_output_length):
        trg_tensor = torch.LongTensor([trg_tokens[-1]]).to(device)
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, context)
        
        # Ambil token dengan probabilitas tertinggi
        top1 = output.argmax(1).item()
        trg_tokens.append(top1)

        # Cek pengulangan token tidak bermakna
        current_token = en_vocab.lookup_token(top1)
        if current_token in [".", "<pad>", "<unk>"]:  # Token yang dianggap tidak bermakna
            if current_token == last_token:
                repeat_count += 1
            else:
                repeat_count = 1
            if repeat_count >= max_repetition:
                break  # Hentikan jika pengulangan melebihi batas
        else:
            repeat_count = 0  # Reset jika token bermakna
        
        last_token = current_token

        # Hentikan jika menemukan <eos>
        if top1 == en_vocab[eos_token]:
            break

    # Konversi ke string
    translated_tokens = en_vocab.lookup_tokens(trg_tokens[1:-1])  # Hapus <sos> & <eos>
    translated_sentence = " ".join(translated_tokens)
    
    return translated_sentence



