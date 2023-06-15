# WaffleCLIP [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

__Authors:__ [Karsten Roth](https://karroth.com/), [Jae Myung Kim](https://jaemyung-kim.github.io/), [Almut Sophia Koepke](https://www.eml-unitue.de/people/almut-sophia-koepke), [Oriol Vinyals](https://scholar.google.com/citations?user=NkzyCvUAAAAJ&hl=en), [Cordelia Schmid](https://scholar.google.com/citations?user=IvqCXP4AAAAJ&hl=en), [Zeynep Akata](https://www.eml-unitue.de/people/zeynep-akata)

---

**Table of Contents**

- [Waffling for Performance ](#waffling-for-performance-)
  - [Overview](#overview)
  - [Setting Up](#setting-up)
  - [Replicating](#replicating)
    - [Default Zero-Shot Visual Classification Performance of CLIP](#default-zero-shot-visual-classification-performance-of-clip)
    - [Utilizing GPT Descriptors](#utilizing-gpt-descriptors)
    - [Generating your own GPT Descriptors](#generating-your-own-gpt-descriptors)
    - [`Waffle`CLIP](#waffleclip)
    - [Utilizing High-Level Concept Guidance](#utilizing-high-level-concept-guidance)
    - [Extract your own High-Level Concepts](#extract-your-own-high-level-concepts)
    - [Putting Everything Together](#putting-everything-together)
  - [Repository Details](#repository-details)
  - [Citation](#citation)

---

## Overview

<!-- <img align="left" style="padding: 0px 20px 0px 0px;"  width="400" padding="100" src=images/teaser.png> -->

This repository contains code to replicate key experiments from our paper [Waffling around for Performance: Visual Classification with Random Words and Broad Concepts](https://arxiv.org/abs/2306.07282).
It should also provide a good starting point for any subsequent research looking to study improved (zero-shot) transfer performance of pretrained Vision Language Models (VLM), and extends the [great repository](https://github.com/sachit-menon/classify_by_description_release) associated with the original [Visual Classification via Description from Large Language Models](https://arxiv.org/abs/2210.07183) paper.

If you find this repository useful or use it as part of your research, please consider [citing it](#citation).

<img width="80%" src=images/main.png>

---

## Setting Up

**Set up environment:** To get started, simply set up the correct environment using `environment.yaml` by running

```bash
conda env create -f environment.yaml
```

and activate the environment via `conda activate waffle`.

**Ensure `clip` is up-to-date:** The above command should install all relevant libraries. If you are not able to utilize the `ViT-L/14` backbone, it is like because your version of `clip` is not up-to-date. In this case, consider re-installing it:

```bash
pip install git+https://github.com/openai/CLIP.git
```

**Downloading datasets:** The associated datasets should download automatically with the exception of `ImageNet`, which should follow the default `ImageNet2012`-structure. We also note that auto-downloading `Places365` sometimes causes some issues, and may need to be downloaded by hand.

---

## Replicating

In this section, we will detail how to run both baseline approaches (CLIP, CLIP + GPT Descriptors), as well as `Waffle`CLIP and its variants. We also showcase how to generate your own descriptions and extract your own high-level concepts to extend default prompts with.

A large collection of sample runs to replicate baseline experiments are provided in `replicate_key_results.sh`, which will create a selection of result csv-files in a `results`-folder. You should be able to extract the information in a more readable fashion by simply using `evaluate_results.py`.

In the following part, we will provide a few more details on specific settings and how to run them.

### Default Zero-Shot Visual Classification Performance of CLIP

To replicate the zero-shot classification performance of vanilla CLIP on e.g. the `ImageNet1K` test data, simply run

```bash
python base_main.py --savename='baselines' --dataset=imagenet --mode=clip --model_size=ViT-B/32
```

which will utilize the `ViT-B/32` backbone for `mode=clip` on `dataset=imagenet`. Generated results are then *appended* to a csv-file named `results/baselines.csv`. To get results for multiple datasets, simply run with respective changes in `--dataset`. A list of available datasets is provided in `waffle_tools.DATASETS`.

### Utilizing GPT Descriptors

To extend the zero-shot classification of vanilla CLIP with GPT-3 generated descriptions following [Menon et al. 2023](https://arxiv.org/abs/2210.07183) on e.g. the `ImageNet1K` test data, simply run

```bash
python base_main.py --savename='baselines_gpt' --dataset=imagenet --mode=gpt_descriptions --model_size=ViT-B/32
```

Generated results are then *appended* to a csv-file named `results/baselines_gpt.csv`.

### Generating your own GPT Descriptors

If you want to produce new GPT Descriptors for other datasets, simply utilize `generate_descriptors.py`, which is adapted from [Menon et al. 2023](https://arxiv.org/abs/2210.07183). Ensure that you have a valid OpenAI account.

### `Waffle`CLIP

To perform zero-shot classification using default `Waffle`CLIP, simple run

```bash
python base_main.py --savename='waffleclip' --dataset=imagenet --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-B/32
```

which utilizes 15 pairs comprising a random word and a random character sequence descriptor (i.e. 30 descriptors in total) for `Waffle`CLIP. The results are computed over 7 different random initializations, and then averaged. Mean and standard deviations are then stored in `results/waffleclip.csv`.

### Utilizing High-Level Concept Guidance

Using high-level concept-guidance is as easy as using zero-shot vanilla CLIP. Given some high-level concept, e.g. `food` for the Food101 dataset, simply run

```bash
python base_main.py --savename='baselines_concept' --dataset=food101 --mode=clip --model_size=ViT-B/32 --label_before_text='A photo of a food: a '
```

which replaces the default prompt primer (`"A photo of a "`) with `"A photo of a food: a "`. This can similarly be applied to e.g. `Waffle`CLIP as shown above by also simply appending and changing the `--label_before_text` parameter.

### Extract your own High-Level Concepts

Given a dataset with classnames, extracting of shared concepts can be simply done using `generate_concepts.py`, which selects a random subset and queries GPT-3 about commonalities.

### Putting Everything Together

To run `Waffle`CLIP on top of GPT-Descriptors and with high-level concept guidance, one can simply combine the commands above and run

```bash
python base_main.py --savename='waffleclip_gpt_concepts' --dataset=food101 --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-B/32 --label_before_text='A photo of a food: a '
```

---

## Repository Details

In this section, we quickly details the implemented CLIP and `Waffle`CLIP variants. Note that all of these methods, except for the notion of high-level concept guidance, are implemented in `waffle_tools.py > load_gpt_descriptions()`.

As baseline methods (executable via `--mode=<name>`), we have

- `clip`: Default vanilla CLIP.
- `gpt_descriptions`: Extends CLIP with GPT-descriptions per class available in the folder `descriptors`.

For randomization studies, we have

- `(shared_)random_descriptions`: Randomly shuffle and redistribute available descriptions to each class, either uniquely or shared between classes (`shared`). By default, a `--randomization_budget=1` means that we use the same number of descriptors per class as is the average for GPT-3 provided descriptors.
- `swapped_descriptions`: Randomly interchange lists of descriptions between classes.
- `scrambled_descriptions`: For lists of descriptions **per** class, we randomly shuffle words and word orders.

For `Waffle`CLIP variants, we have

- `waffle`: Denotes the standard `Waffle`CLIP setup using pairs of random word and random character sequence descriptors. Use `--waffle_count` for the number of pairs.
- `waffle_and_gpt`: Extends `Waffle`CLIP with associated GPT-3 descriptors. Note that the same set of descriptors are used for each class, while the descriptors **per** class may differ. Use the same `--waffle_count` number as in `--mode=waffle`, it will be internally adapted to ensure the same number of descriptors and an equal balanced between randomized ones and GPT-3 descriptions. Note that through the subsampling, resulting random descriptor lists may slightly vary between classes.

For additional CLIP uses, we have also included

- `prompt_ensemble`: Takes `--waffle_count * 2` of prompts randomly sampled from `waffle_tools.prompt_ensemble` and provides a list of multiple prompts to average retrieval over instead. 
- `context_prompt_ensemble`: Extends prompt ensembling with high-level concept guidance. Note that the `--label_before_text` parameter still has to include the extracted high-level concepts.

---

## Citation

```bibtex
@misc{roth2023waffling,
      title={Waffling around for Performance: Visual Classification with Random Words and Broad Concepts}, 
      author={Karsten Roth and Jae Myung Kim and A. Sophia Koepke and Oriol Vinyals and Cordelia Schmid and Zeynep Akata},
      year={2023},
      eprint={2306.07282},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
