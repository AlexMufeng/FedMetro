# FedMetro: Efficient Metro Passenger Flow Prediction via Federated Graph Learning
Official PyTorch implementation of "FedMetro: Efficient Metro Passenger Flow Prediction via Federated Graph Learning"

## Data

We collated and provided three real metro data sets from Beijing (BJMetro), Shanghai (SHMetro) and Hangzhou (HZMetro).



## Environment

Our experiments are implemented in PyTorch 1.13.1 on Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz and four NVIDIA A100 GPUs with 40GB memory.

```bash
conda create -n fedmetro "python=3.11"
conda activate fedmetro
bash install.sh
```



## Run

```bash
bash run.sh
```

