# FiD-NER
Official Code of "Improving Chinese Named Entity Recognition by Search Engine Augmentation"

### Requirements
1. torch=1.10.2+cu113

2. transformers=4.17

3. running with CUDA 11.4 and Python 3.9.7

### How to run this code
1. download BERT pretrained weights from huggingface to `prev_trained_model` directory. We use [bert-base chinese](https://huggingface.co/bert-base-chinese)

2. modify config parameters in scripts/run_ner_crf_xxx.sh

3. run `sh scripts/run_ner_crf_xxx.sh`

### Dataset description
1. Experiments were conducted on three public datasets ([Chinese Resume](https://aclanthology.org/P18-1144/) / [People's Daily](https://icl.pku.edu.cn/) / [Weibo NER](https://aclanthology.org/D15-1064/)) and our self-built unconverntional NER datasets from social media BiliBili.
2. "Unconventional" means named entities contains more polysemous words and grammatical ambiguities. Please contact me if you are interested about this dataset. 
### Citation
If you find FiD-NER interesting and helps your research, please consider citing our work
```
@article{mao2022improving,
  title={Improving Chinese Named Entity Recognition by Search Engine Augmentation},
  author={Mao, Qinghua and Li, Jiatong and Meng, Kui},
  journal={arXiv preprint arXiv:2210.12662},
  year={2022}
}
```
