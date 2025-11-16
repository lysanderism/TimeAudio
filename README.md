# 🕥 TimeAudio: Bridging Temporal Gaps in Large Audio-Language Models


[![arXiv](https://img.shields.io/badge/arXiv-1.1-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/.)
[![Project Page](https://img.shields.io/badge/GitHub.io-Project-blue?logo=Github&style=flat-square)](https://github.com/lysanderism/TimeAudio)
[![🤗 Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/lysanderism/TimeAudio)
[![🤗 Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DataSet-blue)](https://huggingface.co/datasets/lysanderism/FTAR)

---

**This is the official repository for "[TimeAudio: Bridging Temporal Gaps in Large Audio-Language Models](https://arxiv.org/pdf/.)".**


## ✨ Abstract

Recent Large Audio-Language Models (LALMs) exhibit impressive capabilities in understanding audio content for conversational QA tasks. However, these models struggle to accurately understand timestamps for temporal localization (e.g., Temporal Audio Grounding) and are restricted to short audio perception, leading to constrained capabilities on fine-grained tasks. We identify three key aspects that limit their temporal localization and long audio understanding: (i) timestamp representation, (ii) architecture, and (iii) data. To address this, we introduce TimeAudio, a novel method that empowers LALMs to connect their understanding of audio content with precise temporal perception. Specifically, we incorporate unique temporal markers to improve time-sensitive reasoning and apply an absolute time-aware encoding that explicitly grounds the acoustic features with absolute time information. Moreover, to achieve end-to-end long audio understanding, we introduce a segment-level token merging module to substantially reduce audio token redundancy and enhance the efficiency of information extraction. Due to the lack of suitable datasets and evaluation metrics, we consolidate existing audio datasets into a new dataset focused on temporal tasks and establish a series of metrics to evaluate the fine-grained performance. Evaluations show strong performance across a variety of fine-grained tasks, such as dense captioning, temporal grounding, and timeline speech summarization, demonstrating TimeAudio's robust temporal localization and reasoning capabilities.


<!-- ## ✨ Teaser

<p align="center">
  <img src="https://github.com/user-attachments/assets/ea723225-f9c8-4ca2-8837-2c2c08189bdd" alt="method">
</p>
<p style="text-align: left;">(a) Overview of AudioX, illustrating its capabilities across various tasks. (b) Radar chart comparing the performance of different methods across multiple benchmarks. AudioX demonstrates superior Inception Scores (IS) across a diverse set of datasets in audio and music generation tasks.</p> -->


## ✨ Method

<p align="center">
  <img src="./resource/overview.png" alt="method">
</p>
<p align="center">Overview of the TimeAudio Framework.</p>



## Code


### 🛠️ Environment Setup

```bash
git clone https://github.com/lysanderism/TimeAudio
cd TimeAudio
conda create -n TimeAudio python=3.10.16
conda activate TimeAudio
pip install -r requirements.txt
conda install -c conda-forge ffmpeg libsndfile
```

## 🪄 Pretrained Checkpoints

Download the pretrained model from 🤗 [Hugging Face](https://huggingface.co/lysanderism/TimeAudio):


1. Download [whisper large v2](https://huggingface.co/openai/whisper-large-v2/tree/main) to ```whisper_path```.
2. Download [Fine-tuned BEATs_iter3+ (AS2M) (cpt2)](https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D) to `beats_path`.
3. Download [vicuna 7B v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5/tree/main) to ```vicuna_path```.
4. Download [TimeAudio](https://huggingface.co/lysanderism/TimeAudio) on Hugging Face to ```ckpt_path```
5. Put these checkpoints to the path ./pretrained_model/


### 🤗 Cli Inference

To inference the TimeAudio, run:

```bash
ppython3 cli_inference.py --cfg-path configs/infer_config.yaml
```


### 🎯 Data Examples

| Task                 | `AvgDuration`       | `Prompt`                                 | 
|:---------------------|:-------------------|:----------------------------------------------|
| Dense_Audio_Captioning (DAC)  | `11.3s`             | `"What are the sound events? Provide their time intervals and brief descriptions."` |
| Temporal_Audio_Grounding (TAG)  | `9.8s`           | `"What are the start and end times of audio matching 'dogs barking'?",` |
| Timeline_Speech_Summarization	(TSS) | `81.7s` | `"Could you analyze the speech and generate a summary with start and end timestamps for each segment?"` |
| Audio TQA	(TQA) | `10s` | `"What is the total count of the generic impact sounds in the provided audio clip?"`            |

### 🖥️ Train Script

```bash
## stage 1
torchrun --nproc_per_node=8 train.py --cfg-path "configs/config_stage1.yaml"
## stage 2
torchrun --nproc_per_node=8 train.py --cfg-path "configs/config_stage2.yaml"

```

### 🖥️ Batch Inference

```bash
torchrun --nproc_per_node=8 inference.py --cfg-path "configs/infer_config.yaml"
```

## 🚀 Citation
If you find TimeAudio useful, please cite our paper:

```
@article{,
  title={TimeAudio: Bridging Temporal Gaps in Large Audio-Language Models},
  author={Hualei Wang, Yiming Li, Shuo Ma, Hong Liu, Xiangdong Wang},
  journal={arXiv preprint arXiv:},
  year={2025},
  url={https://arxiv.org/abs/}
}
```

## 🙏 Acknowledgments

We gratefully acknowledge the creators of:

- SALMONN(Tang et al.)
- Qwen2-Audio(Chu et al.)

## License

Please follow [CC-BY-NC](./LICENSE).
