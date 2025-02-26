from onnxruntime import InferenceSession
from config import token2idx, idx2tag
import numpy as np
import os

_model_path = os.path.join(os.path.dirname(__file__), "model.onnx")
_model = None

def tagger(text: str, model=None):
  global _model
  if model is None:
    if _model is None:
      _model = InferenceSession(_model_path)

    model = _model

  chars = [c for c in text]
  input_ids = np.array(
    [token2idx[c] if c in token2idx else token2idx["[unk]"] for c in text]
  )

  input_ids = np.expand_dims(input_ids, axis=0)
  outputs = model.run(None, {"input": input_ids})[0]
  tag_ids = np.argmax(outputs, axis=-1).squeeze().tolist()
  tags = [idx2tag[idx] for idx in tag_ids]

  results = []
  i = 0

  for tag, token in zip(tags, chars):
    if tag.startswith("B-") or i == 0:
      results.append((token, tag[2:]))
    else:
      results[-1] = (results[-1][0] + token, results[-1][1])
    i += 1

  return results
