import torch.nn as nn


class Tagger(nn.Module):
  def __init__(
    self,
    n_vocab: int,
    n_classes: int,
    embed_dim,
    hidden_size: int,
    n_layers: int,
    padding_idx: int,
    dropout: float = 0.3,
  ):
    super().__init__()
    self.embedding = nn.Embedding(
      n_vocab,
      embed_dim,
      padding_idx=padding_idx,
    )
    self.embed_norm = nn.LayerNorm(embed_dim)
    self.lstm = nn.LSTM(
      embed_dim,
      hidden_size=hidden_size,
      num_layers=n_layers,
      batch_first=True,
      bidirectional=True,
      dropout=dropout if n_layers > 1 else 0,
    )
    self.lstm_norm = nn.LayerNorm(hidden_size * 2)
    self.dropout = nn.Dropout(dropout)
    self.fc = nn.Linear(hidden_size * 2, n_classes)

  def forward(self, x):
    embs = self.embedding(x)
    embs = self.embed_norm(embs)
    outputs, _ = self.lstm(embs)
    outputs = self.lstm_norm(outputs)
    outputs = self.dropout(outputs)
    outputs = self.fc(outputs)
    return outputs

if __name__ == "__main__":
  import torch

  x = torch.randint(0, 100, (1, 30))
  model = Tagger(
    n_vocab=100,
    n_classes=10,
    embed_dim=128,
    hidden_size=512,
    n_layers=2,
    padding_idx=0,
    dropout=0.1,
  )

  out = model(x)
  print(out.shape)
