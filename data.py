import torch
import re
import random
from torch.utils.data import Dataset
from config import token2idx, tag2idx

_re_delimiter = re.compile(r"[~_\^\u200b]")


def get_values(p: str):
  p = [x.split("/") for x in p.split()]
  p = [(_re_delimiter.sub("", x[0]), x[1]) for x in p]
  return p


class TaggerDataset(Dataset):
  def __init__(self, file, max_seq_len, sampling=False):
    super().__init__()
    self.max_seq_len = max_seq_len
    aug_type = ["sub", "del", "ins", None]

    # load data from the text file
    values = []

    with open(file) as infile:
      self.values = []
      for line in infile:
        line = line.rstrip("\n")
        if not line:
          continue

        for w, t in get_values(line):
          aug = random.choice(aug_type) if sampling else None

          # full word mask
          if sampling:
            if random.randint(0, 100) == 1:
              continue

          for i, c in enumerate(w):            
            if aug is not None and i > 0:
              aug = None

              is_vowel = ord(c) >= 0x17B6 and ord(c) <= 0x17D1
              is_consonant = ord(c) >= 0x1780 and ord(c) <= 0x17B3

              if is_vowel:
                if aug == "sub":
                  c = chr(random.randint(0x17B6, 0x17D1))

                if aug == "del":
                  continue

              if is_consonant:
                if aug == "sub":
                  c = chr(random.randint(0x1780, 0x17B3))

                if aug == "del":
                  continue

                if aug == "ins":
                  values.append(
                    [
                      token2idx[c] if c in token2idx else token2idx["[unk]"],
                      tag2idx[("B-" if i == 0 else "I-") + t],
                      i,
                    ]
                  )

            values.append(
              [
                token2idx[c] if c in token2idx else token2idx["[unk]"],
                tag2idx[("B-" if i == 0 else "I-") + t],
                i,
              ]
            )

    # chunking
    offset = 0
    chunks = []
    alpha = 0

    while offset < len(values):
      seek = offset + self.max_seq_len - alpha

      if sampling:
        alpha = random.randint(0, max_seq_len // 2)

      if seek < len(values):
        while values[seek][-1] != 0:
          seek -= 1
        chunk = values[offset:seek]
        # padding
        if len(chunk) < self.max_seq_len:
          chunk = chunk + ([[0, 0, -1]] * (self.max_seq_len - len(chunk)))
        chunks.append(chunk)
        offset = seek
        continue
      break

    chunk = values[offset:]
    if len(chunk) > 0:
      # padding
      if len(chunk) < self.max_seq_len:
        chunk = chunk + ([[0, 0, -1]] * (self.max_seq_len - len(chunk)))
      chunks.append(chunk)
    self.items = chunks

  def __getitem__(self, index):
    item = self.items[index]
    values = torch.LongTensor(item).T
    input_ids = values[0]
    target_ids = values[1]
    return input_ids, target_ids

  def __len__(self):
    return len(self.items)


if __name__ == "__main__":
  dataset = TaggerDataset("data/train.txt", 64, sampling=True)
  print(f"{len(dataset)=}")
  print(dataset[0])
  print(dataset[0][0].shape)
  print(dataset[0][1].shape)
