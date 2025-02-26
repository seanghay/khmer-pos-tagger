import torch
import torch.nn as nn
from model import Tagger
from data import TaggerDataset
from torch.utils.data import DataLoader
from config import token2idx, tag2idx
from train import evaluate

if __name__ == "__main__":
  max_sequence_length = 64
  device = "mps"
  n_vocab = len(token2idx)
  n_classes = len(tag2idx)
  pad_idx = token2idx["[pad]"]

  model = Tagger(
    n_vocab=n_vocab,
    n_classes=n_classes,
    embed_dim=256,
    hidden_size=384,
    n_layers=2,
    padding_idx=pad_idx,
    dropout=0,
  ).to(device)

  model.load_state_dict(torch.load("ckpt/best.pt", weights_only=True))
  model.eval()

  val_dataset = TaggerDataset("data/val.txt", max_sequence_length)
  val_data_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    drop_last=False,
  )

  criterion = nn.CrossEntropyLoss()
  metrics = evaluate(model, val_data_loader, device, criterion, 0)
  print(metrics)
