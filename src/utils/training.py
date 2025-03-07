import torch
import torch.nn as nn
import torch.optim as optim
from models.gru_seq2seq import Encoder, Decoder, Seq2Seq

def init_model(input_dim, output_dim, embedding_dim, hidden_dim, dropout, pad_index, device):
    encoder = Encoder(input_dim, embedding_dim, hidden_dim, dropout)
    decoder = Decoder(output_dim, embedding_dim, hidden_dim, dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)

    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.01)

    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

    return model, optimizer, criterion

def train_fn(model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device):
    model.train()
    epoch_loss = 0
    for batch in data_loader:
        src = batch["id_ids"].to(device)
        trg = batch["en_ids"].to(device)
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            src = batch["id_ids"].to(device)
            trg = batch["en_ids"].to(device)
            output = model(src, trg, 0)  # Teacher forcing = 0 for evaluation
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)
