# Video Classification on COIN Dataset
This recipe is also explained in [Medium Blog](https://medium.com/@yozkose3/video-classification-with-pytorch-9b7d76776881). 

## Introduction
In this recipe, we will classify cooking and decoration video clips with Pytorch.

## Dataset
I selected 2 categories from [COIN dataset](https://coin-dataset.github.io). There are also sub-categories in primary categories. Selected categories:

<b>Cooking:</b> MakeSandwich, CookOmelet, MakePizza, MakeYoutiao, MakeBurger, MakeFrenchFries
<b>Decoration:</b> AssembleBed, AssembleSofa, AssembleCabinet, AssembleOfficeChair

It is very easy to train with different data setups. You should only change `target_label_list` variable. Default configuration is:
```
target_label_list = [
    ["MakeSandwich", "CookOmelet", "MakePizza", "MakeYoutiao", "MakeBurger", "MakeFrenchFries"],
    ["AssembleBed", "AssembleSofa", "AssembleCabinet", "AssembleOfficeChair"],
]
```
There are 2 lists in `target_label_list` which are classes to classify. Internal lists contain actions in taxonomy.xlsx.


## How to Run
You may changes a few lines in `train.py`
1. If you want, change `target_label_list` to set data classes.
2. If you want, change experiment path `logger = Logger(exp_path="exps/exp1")`
3. Change number of out dimension in `model.blocks[5].proj = nn.Linear(2048, 2).to(device)` according to step 1.
4. RUN `python train.py`