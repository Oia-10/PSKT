# PSKT(MM24)
Source code and datasets for our paper (recently accepted in MM24): [Remembering is Not Applying: Interpretable Knowledge Tracing for Problem-solving Processes](https://doi.org/10.1145/3664647.3681049)

## Dataset
- [ASSIST12](https://sites.google.com/site/assistmentsdata/2012-13-school-data-withaffect)
- [ASSIST17](https://sites.google.com/view/assistmentsdatamining/dataset)
- [junyi](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198)
- [Ednet](https://github.com/riiid/ednet)
- [Eedi](https://eedi.com/projects/neurips-education-challenge)

## Requirements
- pytorch

## Preparation
The data **does not** need to be processed into the common three-row dataset format; a tabular CSV file is sufficient. It should include features such as user_id, problem_id, skill_id,correct, and **time_stamp**. Below is an example:

|user_id|problem_id|skill_id|correct|time_stamp|
|--|--|--|--|--|
0|24796|78|1|1348033800.0
0|24797|78|1|1348033800.0
0|22269|166|0|1348033800.0
0|22279|166|0|1348034200.0
0|22310|167|1|1348034200.0


## Usage
- Step1: download the dataset (the dataset used in this example is `ASSIST17`), then put it in the folder `data/assist17`.

- Step2: run `data/data_pro.py` to preprocess the dataset (the sequence length used in this example is 100).

- Step3: Training.
```shell
python Q5_train.py --dataset assist17 --length 100 --batch_size 64 --q_num 2490 --kc_num 97 --cv_num 0
```
## References
|Method|Paper|Code|
|--|--|--|
PKT | https://doi.org/10.1016/j.eswa.2023.122280| https://github.com/WeiMengqi934/PKT
FKT | https://doi.org/10.1016/j.eswa.2023.122107 | https://github.com/ccnu-edm/FKT
ATDKT|https://doi.org/10.1145/3543507.3583866|https://github.com/pykt-team/pykt-toolkit
LPKT|https://doi.org/10.1145/3447548.3467237|https://github.com/bigdata-ustc/EduKTM
SAINT|https://doi.org/10.1145/3386527.3405945|https://github.com/Shivanandmn/SAINT_plus-Knowledge-Tracing-
AKT|https://doi.org/10.1145/3394486.3403282|https://github.com/arghosh/AKT
DKT-F|https://doi.org/10.1145/3308558.3313565|https://github.com/THUwangcy/HawkesKT/blob/main/src/models/DKTForgetting.py
Deep-IRT|https://doi.org/10.48550/arXiv.1904.11738|https://github.com/ckyeungac/DeepIRT
SAKT|https://doi.org/10.48550/arXiv.1907.06837| https://github.com/arshadshk/SAKT-pytorch
DKVMN|https://doi.org/10.1145/3038912.3052580|https://github.com/jennyzhang0215/DKVMN
DKT|https://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf|https://github.com/chrispiech/DeepKnowledgeTracing

## Citation
If you find this project helpful in your research or work, please consider citing it. Here's the citation format:

```
@inproceedings{PSKT,
    author = {Huang, Tao and Ou, Xinjia and Yang, Huali and Hu, Shengze and Geng, Jing and Hu, Junjie and Xu, Zhuoran},
    title = {Remembering is Not Applying: Interpretable Knowledge Tracing for Problem-solving Processes},
    year = {2024},
    isbn = {9798400706868},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3664647.3681049},
    doi = {10.1145/3664647.3681049},
    booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
    pages = {3151–3159},
    numpages = {9},
    keywords = {distance education, knowledge tracing, problem solving},
    location = {Melbourne VIC, Australia},
    series = {MM '24}
}
```
