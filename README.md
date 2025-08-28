# BMIP:  Bi-directional Modality Interaction Prompt Learning for VLM [IJCAI 2025]



> [**BMIP:  Bi-directional Modality Interaction Prompt Learning for VLM**](https://arxiv.org/abs/2501.07769)<br>[Song-Lin Lv](https://arxiv.org/search/cs?searchtype=author&query=Lv,+S), [Yu-Yang Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen,+Y), [Zhi Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou,+Z), [Ming Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang,+M), [Lan-Zhe Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo,+L)


Official implementation of the paper "[**BMIP:  Bi-directional Modality Interaction Prompt Learning for VLM**](https://arxiv.org/abs/2501.07769)".

<hr />

# :rocket: News
* **(Aug 16, 2025)**
  * Paper accepted at IJCAI 2025 :tada: 
* **(Aug 14, 2025)** 
  * The repository also supports
    [CoOp](configs/trainers/CoOp),
    [Co-CoOp](configs/trainers/CoCoOp),
    [Deep Vision Prompting](configs/trainers/VPT/vit_b16_c2_ep5_batch4_4.yaml),
    [Deep Language Prompting](configs/trainers/IVLP/vit_b16_c2_ep5_batch4_4ctx_language_only.yaml), and 
    [Independent V-L Prompting](configs/trainers/IVLP/vit_b16_c2_ep5_batch4_2+2ctx.yaml)
    architectures.

<hr />

## Highlights

![main figure](docs/main_figure.png)
> **<p align="justify"> Abstract:** *Vision-language models (VLMs) have exhibited remarkable generalization capabilities, and prompt learning for VLMs has attracted great attention for the ability to adapt pre-trained VLMs to specific downstream tasks. However, existing studies mainly focus on single-modal prompts or uni-directional modality interaction, overlooking the powerful alignment effects resulting from the interaction between the vision and language modalities. To this end, we propose a novel prompt learning method called Bi-directional Modality Interaction Prompt (BMIP), which dynamically weights bi-modal information through learning the information of the attention layer, enhancing trainability and inter-modal consistency compared to simple information aggregation methods. To evaluate the effectiveness of prompt learning methods, we propose a more realistic evaluation paradigm called open-world generalization complementing the widely adopted cross-dataset transfer and domain generalization tasks. Comprehensive experiments on various datasets reveal that BMIP not only outperforms current state-of-the-art methods across all three evaluation paradigms but is also flexible enough to be combined with other prompt-based methods for consistent performance enhancement.* </p>

## Main Contributions

1) Novel Bi-directional Modality Interaction Technique
   * Enhance the cross-modality alignment and pave the way for further exploration of information aggregation in other multi-modal modelsNew 
2) Evaluation Paradigm: Open-World Generalization
   * Facilitate more realistic evaluations and promote related research 
3) Flexible Integration with Other Methods
   * BMIP is flexible enough to combine with other prompt learning methods, consistently boosting their performance.
4) State-of-the-Art Performance
   * BMIP achieves SOTA performance across all tasks


## :ballot_box_with_check: Supported Methods

| Method                    | Paper                                         |                           Configs                            |          Training Scripts          |
| ------------------------- | :-------------------------------------------- | :----------------------------------------------------------: | :--------------------------------: |
| BMIP                      | IJCAI 2025                                    |                [link](configs/trainers/BMIP)                 |        [link](scripts/bmip)        |
| MaPLe                     | [CVPR 2023](https://arxiv.org/abs/2210.03117) | [link](configs/trainers/MaPLeold/vit_b16_c2_ep5_batch4_2ctx.yaml) |       [link](scripts/maple)        |
| CoOp                      | [IJCV 2022](https://arxiv.org/abs/2109.01134) |                [link](configs/trainers/CoOp)                 |        [link](scripts/coop)        |
| Co-CoOp                   | [CVPR 2022](https://arxiv.org/abs/2203.05557) |               [link](configs/trainers/CoCoOp)                |       [link](scripts/cocoop)       |
| Deep Vision Prompting     | -                                             |  [link](configs/trainers/VPT/vit_b16_c2_ep5_batch4_4.yaml)   |        [link](scripts/vpt)         |
| Deep Language Prompting   | -                                             | [link](configs/trainers/IVLP/vit_b16_c2_ep5_batch4_4ctx_language_only.yaml) | [link](scripts/language-prompting) |
| Independent V-L Prompting | -                                             | [link](configs/trainers/IVLP/vit_b16_c2_ep5_batch4_2+2ctx.yaml) |  [link](scripts/independent-vlp)   |

<hr />

## Results
### BMIP in comparison with existing methods
Results reported below show accuracy for base and novel classes for across 11 recognition datasets averaged over 3 seeds.

| Name                                       | Base Acc. | Novel Acc. |    HM     | Epochs |
| ------------------------------------------ | :-------: | :--------: | :-------: | :----: |
| [CLIP](https://arxiv.org/abs/2103.00020)   |   69.34   |   74.22    |   71.70   |   -    |
| [CoOp](https://arxiv.org/abs/2109.01134)   | **82.69** |   63.22    |   71.66   |  200   |
| [CoCoOp](https://arxiv.org/abs/2203.05557) |   80.47   |   71.69    |   75.83   |   10   |
| [MaPLe](https://arxiv.org/abs/2210.03117)  |   82.28   | **75.14**  | **78.55** |   5    |
| [BMIP](https://arxiv.org/abs/2501.07769)   |   83.47   |   76.69    |   79.04   |   10   |

## Installation 
For installation and other package requirements, please follow the instructions detailed in [INSTALL.md](docs/INSTALL.md). 

## Data preparation
Please follow the instructions at [DATASETS.md](docs/DATASETS.md) to prepare all datasets.


## Training and Evaluation
Please refer to the [RUN.md](docs/RUN.md) for detailed instructions on training, evaluating and reproducing the results using our pre-trained models.


<hr />

## Citation
If you use our work, please consider citing:
```bibtex
@misc{lv2025bmipbidirectionalmodalityinteraction,
      title={BMIP: Bi-directional Modality Interaction Prompt Learning for VLM}, 
      author={Song-Lin Lv and Yu-Yang Chen and Zhi Zhou and Ming Yang and Lan-Zhe Guo},
      year={2025},
      eprint={2501.07769},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.07769}, 
}
```

## Contact
If you have any questions, please create an issue on this repository or contact at lvsl@lamda.nju.edu.cn.


## Acknowledgements

Our code is based on [Co-CoOp and CoOp](https://github.com/KaiyangZhou/CoOp) repository. We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.

