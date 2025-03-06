# src/utils/inference.py
import torch

def translate_sentence(sentence, model, tokenizer, en_vocab, id_vocab, sos_token, eos_token, device, max_output_length=25):
    model.eval()
    tokens = [sos_token] + tokenizer.tokenize(sentence.lower())[:1000] + [eos_token]
    numericalized = [id_vocab[token] if token in id_vocab else id_vocab["<unk>"] for token in tokens]
    sentence_tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device)
    with torch.no_grad():
        context = model.encoder(sentence_tensor)
    trg_tokens = [en_vocab[sos_token]]
    hidden = context
    for _ in range(max_output_length):
        trg_tensor = torch.LongTensor([trg_tokens[-1]]).to(device)
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, context)
        top1 = output.argmax(1).item()
        trg_tokens.append(top1)
        if top1 == en_vocab[eos_token]:
            break
    translated_tokens = en_vocab.lookup_tokens(trg_tokens[1:-1])
    return " ".join(translated_tokens)

