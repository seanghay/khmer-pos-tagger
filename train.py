import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Tagger
from config import token2idx, tag2idx
from data import TaggerDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


def evaluate(model, data_loader, device, criterion, epoch):
  eval_loss = 0
  eval_iteration = 0
  eval_batch_iter = tqdm(data_loader, ascii=True)

  # Initialize metric counters
  total_tokens = 0
  correct_tokens = 0

  true_positives = 0
  false_positives = 0
  false_negatives = 0

  all_predictions = []
  all_targets = []

  model.eval()

  with torch.no_grad():
    for input_ids, target_ids in eval_batch_iter:
      input_ids, target_ids = input_ids.to(device), target_ids.to(device)
      logits = model(input_ids)

      target_ids_flat = target_ids.view(-1)
      logits_flat = logits.view(-1, logits.shape[2])

      loss = criterion(logits_flat, target_ids_flat)
      eval_loss += loss.item()
      eval_iteration += 1

      _, predictions = torch.max(logits, dim=-1)

      pad_idx = token2idx["[pad]"]
      mask = target_ids != pad_idx  # Create mask to ignore padding tokens

      correct_tokens += ((predictions == target_ids) & mask).sum().item()
      total_tokens += mask.sum().item()

      for tag_idx in range(1, len(tag2idx)):  # Skip padding class
        true_positives += (
          ((predictions == tag_idx) & (target_ids == tag_idx)).sum().item()
        )

        false_positives += (
          ((predictions == tag_idx) & (target_ids != tag_idx)).sum().item()
        )

        false_negatives += (
          ((predictions != tag_idx) & (target_ids == tag_idx)).sum().item()
        )

      all_predictions.extend(predictions[mask].cpu().numpy())
      all_targets.extend(target_ids[mask].cpu().numpy())

      eval_batch_iter.set_description(
        f"[eval ] epoch: {epoch + 1}, loss: {eval_loss / eval_iteration:.4f}"
      )

  accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
  precision = (
    true_positives / (true_positives + false_positives)
    if (true_positives + false_positives) > 0
    else 0
  )
  recall = (
    true_positives / (true_positives + false_negatives)
    if (true_positives + false_negatives) > 0
    else 0
  )

  f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
  print(f"Evaluation metrics for epoch {epoch + 1}:")
  print(f"  Loss: {eval_loss / eval_iteration:.4f}")
  print(f"  Accuracy: {accuracy:.4f}")
  print(f"  Precision: {precision:.4f}")
  print(f"  Recall: {recall:.4f}")
  print(f"  F1 Score: {f1:.4f}")

  metrics = {
    "loss": eval_loss / eval_iteration,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
  }

  return metrics


def train(model, data_loader, device, optimizer, criterion, epoch):
  train_loss = 0
  train_iteration = 0
  train_batch_iter = tqdm(data_loader, ascii=True)

  model.train()
  for input_ids, target_ids in train_batch_iter:
    input_ids, target_ids = input_ids.to(device), target_ids.to(device)
    logits = model(input_ids)

    # transpose
    target_ids = target_ids.view(-1)
    logits = logits.view(-1, logits.shape[2])

    loss = criterion(logits, target_ids)
    optimizer.zero_grad()

    train_loss += loss.item()
    train_iteration += 1

    loss.backward()
    optimizer.step()

    train_batch_iter.set_description(
      f"[train] epoch: {epoch + 1}, loss: {train_loss / train_iteration:.4f}"
    )


def main():
  lr = 1e-3
  train_batch_size = 16
  eval_batch_size = 16
  max_sequence_length = 64
  n_epoch = 20
  device = "mps"
  n_vocab = len(token2idx)
  n_classes = len(tag2idx)
  pad_idx = token2idx["[pad]"]

  print(
    dict(
      n_vocab=n_vocab,
      n_classes=n_classes,
      device=device,
      train_batch_size=train_batch_size,
      eval_batch_size=eval_batch_size,
      max_sequence_length=max_sequence_length,
      lr=lr,
    )
  )

  train_dataset = TaggerDataset("data/train.txt", max_sequence_length, sampling=True)
  val_dataset = TaggerDataset("data/val.txt", max_sequence_length)

  train_data_loader = DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    shuffle=True,
  )

  val_data_loader = DataLoader(
    val_dataset,
    batch_size=eval_batch_size,
    shuffle=False,
    drop_last=False,
  )

  model = Tagger(
    n_vocab=n_vocab,
    n_classes=n_classes,
    embed_dim=256,
    hidden_size=384,
    n_layers=2,
    padding_idx=pad_idx,
    dropout=0.3,
  ).to(device)

  optimizer = optim.Adam(model.parameters(), lr=lr)
  criterion = nn.CrossEntropyLoss()

  scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1,
    verbose=True,
    min_lr=1e-6,
  )

  os.makedirs("ckpt", exist_ok=True)
  previous_metrics = None

  with open(os.path.join("ckpt", "logs.txt"), "w") as outfile:
    for epoch in range(n_epoch):
      train(model, train_data_loader, device, optimizer, criterion, epoch)
      metrics = evaluate(model, val_data_loader, device, criterion, epoch)
      metrics["epoch"] = epoch + 1
      metrics["time"] = time()

      scheduler.step(metrics["loss"])

      current_lr = optimizer.param_groups[0]["lr"]
      print(f"Current learning rate: {current_lr}")

      if previous_metrics is not None and metrics["loss"] < previous_metrics["loss"]:
        torch.save(model.state_dict(), os.path.join("ckpt", "best.pt"))
        print(f"saved best: {epoch}")
        metrics["best"] = True

      torch.save(model.state_dict(), os.path.join("ckpt", "last.pt"))
      outfile.write(json.dumps(metrics) + "\n")
      outfile.flush()

      previous_metrics = metrics


if __name__ == "__main__":
  main()
