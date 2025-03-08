import torch
import torch.nn as nn
import torch.optim as optim
from models.gru_seq2seq import Encoder, Decoder, Seq2Seq
import tqdm
import numpy as np
import os
import json

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






def train_model(model, train_loader, valid_loader, optimizer, criterion, config, resume_training=False):
    """
    Function to train a sequence-to-sequence model with GRU.
    Supports resuming training from a saved checkpoint.
    """
    epochs = config["training"]["epochs"]
    clip = config["training"]["clip"]
    patience = config["training"]["patience"]
    checkpoint_path = config["training"]["checkpoint_path"]

    teacher_forcing_initial = config["training"]["teacher_forcing_initial"]
    teacher_forcing_final = config["training"]["teacher_forcing_final"]

    best_valid_loss = float("inf")
    patience_counter = 0
    start_epoch = 0
    history = {"train_loss": [], "valid_loss": [], "train_ppl": [], "valid_ppl": []}

    if resume_training and os.path.exists(checkpoint_path):
        print(f"🔄 Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_valid_loss = checkpoint["best_valid_loss"]
        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint["history"]

    model.to(config['training']["device"])

    for epoch in tqdm.tqdm(range(start_epoch, epochs), desc="Training Progress"):
        teacher_forcing_ratio = teacher_forcing_initial - \
                                (teacher_forcing_initial - teacher_forcing_final) * \
                                (epoch / (epochs - 1))

        train_loss = train_fn(model, train_loader, optimizer, criterion, clip, teacher_forcing_ratio, config['training']["device"])
        valid_loss = evaluate_fn(model, valid_loader, criterion, config['training']["device"])

        train_ppl = np.exp(train_loss)
        valid_ppl = np.exp(valid_loss)

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_ppl"].append(train_ppl)
        history["valid_ppl"].append(valid_ppl)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_valid_loss": best_valid_loss,
                "history": history
            }, checkpoint_path)

            print(f"✅ Model saved at epoch {epoch + 1} with valid loss: {valid_loss:.3f}")
        else:
            patience_counter += 1

        print(f"📌 Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Train PPL: {train_ppl:.3f}")
        print(f"📌 Valid Loss: {valid_loss:.3f} | Valid PPL: {valid_ppl:.3f}")

        if patience_counter >= patience:
            print(f"⏹️ Early stopping at epoch {epoch + 1}")
            break

    with open("training_history.json", "w") as f:
        json.dump(history, f)

    print("🏁 Training Finished!")

    return history