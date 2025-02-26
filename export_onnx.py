import torch
from model import Tagger
from config import token2idx, tag2idx, idx2tag

if __name__ == "__main__":
  max_sequence_length = 64
  device = "cpu"
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

  input_ids = torch.randint(0, n_vocab, (1, 128))
  torch.onnx.export(
    model,
    (input_ids,),
    "ckpt/model.onnx",
    export_params=True,
    verbose=True,
    opset_version=15,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
      "input": {0: "batch_size", 1: "seq"},
      "output": {0: "batch_size", 1: "seq"},
    },
  )
