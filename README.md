### Joint Khmer Part-of-Speech Tagger and Word Segmenter

An open-source part of speech tagger for Khmer language using BiLSTM.

The model weights can be downloaded in the release page. 

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
MIT
```


### References

- [khPOS (Khmer Part-of-Speech) Corpus for Khmer NLP Research and Developments](https://github.com/ye-kyaw-thu/khPOS/)
- [Joint Khmer Word Segmentation and Part-of-Speech Tagging Using Deep Learning](https://arxiv.org/abs/2103.16801) 