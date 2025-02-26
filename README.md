### Joint Khmer Part-of-Speech Tagger and Word Segmenter

An open-source part of speech tagger for Khmer language using BiLSTM.

The model weights can be downloaded in the [release page](https://github.com/seanghay/khmer-pos-tagger/releases/tag/0.1). 

Once downloaded, place the model weight at `ckpt/best.pt`

```shell
pip install -r requirements.txt

python infer.py

# Result => [
#   ("ខ្ញុំ", "PRO"),
#   ("ឈ្មោះ", "NN"),
#   ("ថូយ", "PN"),
#   ("ប៊ុន", "PN"),
#   ("ឡូ", "PN"),
#   ("ឆ្នាំ", "NN"),
#   ("នេះ", "DT"),
#   ("វា", "PRO"),
#   ("ត្រូវបាន", "VB"),
#   ("រាយការណ៍", "VB"),
#   ("ថា", "IN"),
#   ("បាន", "AUX"),
#   ("ធ្លាក់", "VB"),
#   ("ចុះ", "RB"),
#   ("មក", "IN"),
#   ("ត្រឹម", "IN"),
#   ("២,២", "CD"),
#   ("ភាគរយ", "NN"),
#   ("។", "KAN"),
#   ("លោក", "PRO"),
#   ("យ៉ាត់", "PN"),
#   ("សៀង", "PN"),
#   ("ហៃ", "PN"),
#   ("ជា", "IN"),
#   ("អ្នកធ្វើ", "NN"),
#   ("វា", "PRO"),
#   ("មែន", "RB"),
#   ("ទេ", "PA"),
#   ("?", "SYM"),
# ]
```

### Library

To make thing easier to access, we made publish a simple library called `khmertagger` which can be installed with

```
pip install khmertagger
```

```python
from khmertagger import tagger

output = tagger("ខ្ញុំឈ្មោះថូយប៊ុនឡូឆ្នាំនេះវាត្រូវបានរាយការណ៍ថាបានធ្លាក់ចុះមកត្រឹម២,២ភាគរយ។លោកយ៉ាត់សៀងហៃជាអ្នកធ្វើវាមែនទេ?")

print(output)
# Result => [
#   ("ខ្ញុំ", "PRO"),
#   ("ឈ្មោះ", "NN"),
#   ("ថូយ", "PN"),
#   ("ប៊ុន", "PN"),
#   ("ឡូ", "PN"),
#   ("ឆ្នាំ", "NN"),
#   ("នេះ", "DT"),
#   ("វា", "PRO"),
#   ("ត្រូវបាន", "VB"),
#   ("រាយការណ៍", "VB"),
#   ("ថា", "IN"),
#   ("បាន", "AUX"),
#   ("ធ្លាក់", "VB"),
#   ("ចុះ", "RB"),
#   ("មក", "IN"),
#   ("ត្រឹម", "IN"),
#   ("២,២", "CD"),
#   ("ភាគរយ", "NN"),
#   ("។", "KAN"),
#   ("លោក", "PRO"),
#   ("យ៉ាត់", "PN"),
#   ("សៀង", "PN"),
#   ("ហៃ", "PN"),
#   ("ជា", "IN"),
#   ("អ្នកធ្វើ", "NN"),
#   ("វា", "PRO"),
#   ("មែន", "RB"),
#   ("ទេ", "PA"),
#   ("?", "SYM"),
# ]
```

### Dataset

The model is trained on [ye-kyaw-thu/khPOS/](https://github.com/ye-kyaw-thu/khPOS/) dataset. See [./data](./data) directory.


### Train

```shell
python train.py
```



### Evaluate

```shell
python evalulate.py
```

### Metrics

Accuracy: `0.9409951303801445`



### License

```
MIT License

Copyright (c) 2025 Seanghay Yath

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


### References

- [khPOS (Khmer Part-of-Speech) Corpus for Khmer NLP Research and Developments](https://github.com/ye-kyaw-thu/khPOS/)
- [Joint Khmer Word Segmentation and Part-of-Speech Tagging Using Deep Learning](https://arxiv.org/abs/2103.16801) 
