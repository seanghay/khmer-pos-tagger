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

  text = "ខ្ញុំឈ្មោះថូយប៊ុនឡូឆ្នាំនេះវាត្រូវបានរាយការណ៍ថាបានធ្លាក់ចុះមកត្រឹម២,២ភាគរយ។លោកយ៉ាត់សៀងហៃជាអ្នកធ្វើវាមែនទេ?"
  chars = [c for c in text]
  input_ids = [token2idx[c] if c in token2idx else token2idx["[unk]"] for c in text]
  input_ids = torch.tensor(input_ids).unsqueeze(dim=0).to(device)

  with torch.no_grad():
    logits = model(input_ids)
    tag_ids = logits.argmax(dim=-1).squeeze().tolist()

  tags = [idx2tag[idx] for idx in tag_ids]

  results = []
  i = 0
  for tag, token in zip(tags, chars):
    if tag.startswith("B-") or i == 0:
      results.append((token, tag[2:]))
    else:
      results[-1] = (results[-1][0] + token, results[-1][1])
    i += 1

  print(results)
