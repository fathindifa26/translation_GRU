import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import torchtext
import tqdm
import sys
import os
# Tambahkan path ke src agar bisa mengakses data/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from data.preprocessing import preprocess_dataset  # ✅ Import ulang setelah menambahkan path
from data.data_loader import get_data_loader
from utils.training import init_model, train_fn, evaluate_fn
from datasets import load_from_disk



# import file
import os
dataset_path = os.path.abspath("../../data/indonesia_jawa_dataset")
print(f"Dataset path: {dataset_path}")

dataset = load_from_disk(dataset_path)
print(dataset)
train_data, valid_data, test_data = (
    dataset["train"],
    dataset["validation"],
    dataset["test"],
)


# Preprocessing
train_data, valid_data, test_data, en_vocab, id_vocab = preprocess_dataset(dataset)


# data loader
pad_index = en_vocab["<pad>"]
batch_size = 128  
train_loader = get_data_loader(train_data, batch_size=batch_size, pad_index=pad_index, shuffle=True)
valid_loader = get_data_loader(valid_data, batch_size=batch_size, pad_index=pad_index, shuffle=False)
test_loader = get_data_loader(test_data, batch_size=batch_size, pad_index=pad_index, shuffle=False)


# TRAIN
input_dim = len(id_vocab)
output_dim = len(en_vocab)
embedding_dim = 512
hidden_dim = 1024
dropout = 0.3
clip = 1.0
teacher_forcing_initial = 0.5
teacher_forcing_final = 0.1
epochs = 10
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model, optimizer, criterion = init_model(input_dim, output_dim, embedding_dim, hidden_dim, dropout, pad_index, device)

# Training Loop
best_valid_loss = float("inf")
patience = 5
patience_counter = 0

for epoch in tqdm.tqdm(range(epochs)):
    teacher_forcing_ratio = teacher_forcing_initial - \
                            (teacher_forcing_initial - teacher_forcing_final) * \
                            (epoch / (epochs - 1))
    
    train_loss = train_fn(model, train_loader, optimizer, criterion, clip, teacher_forcing_ratio, device)
    valid_loss = evaluate_fn(model, valid_loader, criterion, device)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "checkpoints/gru_model.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Train PPL: {np.exp(train_loss):.3f}")
    print(f"Valid Loss: {valid_loss:.3f} | Valid PPL: {np.exp(valid_loss):.3f}")
