# Improving Efficiency in Active Learning based on Data Augmentation Strategies



This is a [PyTorch](https://pytorch.org/) implementation of our thesis.

Title: Improving Efficiency in Active Learning based on Data Augmentation Strategies.



## Start Running IncreVR

### Data Preparation

The datasets included in this paper are all public datasets available from official websites or mirrors. Just put them in the directory as follows.

```
data
├── cifar10
│   ├── cifar-10-batches-py
│   └── cifar-10-python.tar.gz
├── FashionMNIST
│   └── raw
│       ├── t10k-images-idx3-ubyte
│       ├── t10k-images-idx3-ubyte.gz
│       ├── t10k-labels-idx1-ubyte
│       ├── t10k-labels-idx1-ubyte.gz
│       ├── train-images-idx3-ubyte
│       ├── train-images-idx3-ubyte.gz
│       ├── train-labels-idx1-ubyte
│       └── train-labels-idx1-ubyte.gz
├── medical
│   ├── data.csv
│   └── x_10022_y_8184_A2962.png
│   └── ...png


deep-active-learning
├── architectures
├── datasets
├── evaluation
├── main.py
├── query_strategies
├── test.sh
└── utils
```



### Start Running

We provide the following shell codes for model training in `test.sh`. Just following the format provided in `test.sh` when running our codes. We also provide a illustration for training configurations as follows.

1. Basic Active Learning setting
   - `--strategy`: The basic AL strategy to apply, including EntropySampling, LeastConfidence, MarginSampling and BadgeSampling.
   - `--num-init-labels`: The size of initial labeled pool
   - `--n-cycle`: The number of active learning cycles
   - `--num-query`: The number of unlabeled samples to query at each cycle
   - `--updating`: Whether to load the model parameters obtained from the last cycle
   - `--n-epoch`: The number of training epochs for each cycle

2. Setting for augmentation policies
   - `--aug-metric-ulb`: How to aggregate informativeness induced from augmentations, including `min` for score-based strategies (EntropySampling, LeastConfidence, MarginSampling), and `standard` for BADGE strategy.
