from onnxruntime import InferenceSession
import numpy as np
from config import token2idx, tag2idx, idx2tag

if __name__ == "__main__":
  model = InferenceSession("ckpt/model.onnx")
  text = "ខ្ញុំឈ្មោះថូយប៊ុនឡូឆ្នាំនេះវាត្រូវបានរាយការណ៍ថាបានធ្លាក់ចុះមកត្រឹម២,២ភាគរយ។លោកយ៉ាត់សៀងហៃជាអ្នកធ្វើវាមែនទេ?"

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

  print(results)
