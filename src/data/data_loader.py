# src/data/data_loader.py
import torch
import torch.nn as nn

def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_en_ids = [example["en_ids"] for example in batch]
        batch_id_ids = [example["id_ids"] for example in batch]
        batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)
        batch_id_ids = nn.utils.rnn.pad_sequence(batch_id_ids, padding_value=pad_index)
        return {"en_ids": batch_en_ids, "id_ids": batch_id_ids}
    return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
