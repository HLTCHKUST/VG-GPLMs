# Vision Guided Generative Pre-trained Language Models for Multimodal Abstractive Summarization
[![](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/) [![CC BY 4.0][cc-by-shield]][cc-by]

<img align="right" src="img/HKUST.jpg" width="15%"/>

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg


[[Paper]](https://arxiv.org/abs/2109.02401) accepted at the EMNLP 2021:

**Vision Guided Generative Pre-trained Language Models for Multimodal Abstractive Summarization**, by [Tiezheng Yu *](https://tysonyu.github.io), [Wenliang Dai *](https://wenliangdai.github.io/), [Zihan Liu](https://zliucr.github.io/), [Pascale Fung](https://pascale.home.ece.ust.hk).

## Paper Abstract

> Multimodal abstractive summarization (MAS) models that summarize videos (vision modality) and their corresponding transcripts (text modality) are able to extract the essential information from massive multimodal data on the Internet. Recently, large-scale generative pre-trained language models (GPLMs) have been shown to be effective in text generation tasks. However, existing MAS models cannot leverage GPLMs' powerful generation ability. To fill this research gap, we aim to study two research questions: 1) how to inject visual information into GPLMs without hurting their generation ability; and 2) where is the optimal place in GPLMs to inject the visual information?
In this paper, we present a simple yet effective method to construct vision guided (VG) GPLMs for the MAS task using attention-based add-on layers to incorporate visual information while maintaining their original text generation ability.
Results show that our best model significantly surpasses the prior state-of-the-art model by 5.7 ROUGE-1, 5.3 ROUGE-2, and 5.1 ROUGE-L scores on the How2 dataset, and our visual guidance method contributes 83.6\% of the overall improvement.
Furthermore, we conduct thorough ablation studies to analyze the effectiveness of various modality fusion methods and fusion locations.

If you work is inspired by our paper or code, please cite it, thanks!
<pre>
@article{yu2021vision,
  title={Vision Guided Generative Pre-trained Language Models for Multimodal Abstractive Summarization},
  author={Yu, Tiezheng and Dai, Wenliang and Liu, Zihan and Fung, Pascale},
  journal={arXiv preprint arXiv:2109.02401},
  year={2021}
}
</pre>

## Evaluation
We release the generated summaries from different models in ```./evaluation/results```. All the evaluation metrics can be computed following ```./evaluation/README.md```.

## Prepare dataset
You can go to How2 dataset [Github](https://github.com/srvk/how2-dataset) to get the dataset. We recommend you to choose the (option 1): Download a pre-packaged version.

## Run fine-tuning
+ make directory for saving lightning logs: ```mkdir lightning_logs```
+ An example of running Bart text only model: ```./scripts/Bart_text_only.sh```
+ An example of running Bart multimodal model: ```./scripts/Bart_multimodal.sh```

## Run inference
+ An example of running Bart multimodal model: ```./scripts/test_Bart_multimodal.sh```
