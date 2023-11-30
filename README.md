# Towards Real-World Focus Stacking with Deep Learning
### [Paper](https://arxiv.org/abs/2311.17846) | [Dataset](https://drive.google.com/file/d/1aCskAEDjDn2V9t4R6MMLFmNZgMemHdCN/view?usp=sharing) | [Bibtex](#bibtex) 

## Abstract

Focus stacking is widely used in micro, macro, and land- scape photography to reconstruct all-in-focus images from multiple frames obtained with focus bracketing, that is, with shallow depth of field and different focus planes. Existing deep learning approaches to the underlying multi-focus im- age fusion problem have limited applicability to real-world imagery since they are designed for very short image se- quences (two to four images), and are typically trained on small, low-resolution datasets either acquired by light-field cameras or generated synthetically. We introduce a new dataset consisting of 94 high-resolution bursts of raw im- ages with focus bracketing, with pseudo ground truth com- puted from the data using state-of-the-art commercial soft- ware. This dataset is used to train the first deep learn- ing algorithm for focus stacking capable of handling bursts of sufficient length for real-world applications. Qualita- tive experiments demonstrate that it is on par with exist- ing commercial solutions in the long-burst, realistic regime while being significantly more tolerant to noise.

## Bibtex

```
@article{araujo2022focus,
  title={Towards Real-World Focus Stacking with Deep Learning},
  author={Araujo, Alexandre and Ponce, Jean and Mairal, Julien},
  journal={arXiv preprint arXiv:2311.17846},
  year={2022}
}
```
