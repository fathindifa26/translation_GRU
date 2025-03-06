# src/experiments/train_gru.py
# src/experiments/train_gru.py
import sys
import os
# Pastikan src bisa ditemukan
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.optim as optim
import numpy as np
import tqdm
import yaml
from datasets import load_from_disk
from src.data.preprocessing import tokenize_gru, numericalize_example, build_vocabulary
from src.data.data_loader import get_data_loader
from src.models.gru_seq2seq import Encoder, Decoder, Seq2Seq
from src.utils.training import train_fn, evaluate_fn
from src.utils.inference import translate_sentence
from transformers import BertTokenizer

# Load konfigurasi
with open('configs/gru_seq2seq.yaml', "r") as f:
    config = yaml.safe_load(f)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Load dataset
dataset = load_from_disk(config["data"]["path"])
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Preprocessing
train_data = dataset["train"].map(lambda x: tokenize_gru(x, tokenizer))
valid_data = dataset["validation"].map(lambda x: tokenize_gru(x, tokenizer))
test_data = dataset["test"].map(lambda x: tokenize_gru(x, tokenizer))
en_vocab, id_vocab = build_vocabulary(dataset)
train_data = train_data.map(lambda x: numericalize_example(x, en_vocab, id_vocab))
valid_data = valid_data.map(lambda x: numericalize_example(x, en_vocab, id_vocab))
test_data = test_data.map(lambda x: numericalize_example(x, en_vocab, id_vocab))
train_data = train_data.with_format("torch", columns=["en_ids", "id_ids"])
valid_data = valid_data.with_format("torch", columns=["en_ids", "id_ids"])
test_data = test_data.with_format("torch", columns=["en_ids", "id_ids"])

# DataLoader
train_data_loader = get_data_loader(train_data, config["training"]["batch_size"], en_vocab["<pad>"], shuffle=True)
valid_data_loader = get_data_loader(valid_data, config["training"]["batch_size"], en_vocab["<pad>"])
test_data_loader = get_data_loader(test_data, config["training"]["batch_size"], en_vocab["<pad>"])

# Model
encoder = Encoder(
    len(id_vocab),
    config["model"]["embedding_dim"],
    config["model"]["hidden_dim"],
    config["model"]["dropout"]
)
decoder = Decoder(
    len(en_vocab),
    config["model"]["embedding_dim"],
    config["model"]["hidden_dim"],
    config["model"]["dropout"]
)
model = Seq2Seq(encoder, decoder, device).to(device)
model.apply(lambda m: m.weight.data.normal_(0, 0.01) if hasattr(m, "weight") else None)

# Optimizer dan criterion
optimizer = optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss(ignore_index=en_vocab["<pad>"])

# Pelatihan
best_valid_loss = float("inf")
patience = 5
patience_counter = 0

for epoch in tqdm.tqdm(range(config["training"]["epochs"])):
    teacher_forcing_ratio = config["training"]["teacher_forcing_initial"] - \
                            (config["training"]["teacher_forcing_initial"] - config["training"]["teacher_forcing_final"]) * \
                            (epoch / (config["training"]["epochs"] - 1))
    train_loss = train_fn(model, train_data_loader, optimizer, criterion, config["training"]["clip"],
                          teacher_forcing_ratio, device)
    valid_loss = evaluate_fn(model, valid_data_loader, criterion, device)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "gru_model.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Train PPL: {np.exp(train_loss):.3f}")
    print(f"Valid Loss: {valid_loss:.3f} | Valid PPL: {np.exp(valid_loss):.3f}")

# Evaluasi
model.load_state_dict(torch.load("gru_model.pt"))
test_loss = evaluate_fn(model, test_data_loader, criterion, device)
print(f"Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):.3f}")

# Contoh inferensi
translated = translate_sentence("aku akan makan", model, tokenizer, en_vocab, id_vocab, "<sos>", "<eos>", device)
print(f"Translated: {translated}")
