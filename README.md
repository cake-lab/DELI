# DELI 🥪

Nicholas Krichevsky, Matthew St Louis, Tian Guo

This is the official repository associated with our paper: Quantifying and Improving Performance of Distributed Deep Learning with Cloud Storage, to be published in [IC2E 2021](https://conferences.computer.org/IC2E/2021/).

## Overview 

DELI 🥪 is a PyTorch-based prototype for enabling efficient distributed deep learning using cloud storage buckets. Specifically, DELI allows trainining data to be stored using storage buckets and leverages two classical techniques, namely caching and pre-fetching, to mitigate the training performance degradation. We evaluated the training performance of two deep learning workloads using Google Cloud’s NVIDIA K80 GPU servers and show that we can reduce the time that the training loop is waiting for data by 85.6% - 93.5% compared to loading from a storage bucket—thus achieving comparable performance to loading data directly from disk while only storing a fraction of the data locally at a time. In addition, this has the potential of lowering the cost of running a training workload, especially on models with long per-epoch training times.

## Directory Structure

- `src` includes the various components used to implement and test DELI.
- `data` includes the raw data collected for the purposes of the paper.
- `notebooks` includes the Jupyter Notebooks used to create the figures in the paper.

Each directory has its own relevant README to help guide setup; these are generally fairly complete, but not necessarily exact in the steps needed for setup. The authors encourage anyone with questions to reach out or to file an issue. Certain oddities that were in place for our testing are left in, simply to ensure the setup used is known.

## Paper 

[Quantifying and Improving Performance of Distributed Deep Learning with Cloud Storage]()

If you use DELI's data or code, please cite: 

```bibtex
@InProceedings{DELI_IC2E2021,
    title = "{Quantifying and Improving Performance of Distributed Deep Learning with Cloud Storage}",
    booktitle = {International Conference on Cloud Engineering},
    series = {IC2E '21},
    author="Krichevsky, Nicholas and St Louis, Matthew and Guo, Tian",
    year="2021",
}
```

## Acknowledgement

This work was conducted as part of WPI's CS Major Qualifying Project and is supported in part by National Science Foundation grants CNS-#1755659 and CNS-#1815619, WPI CS Department funding, and Google Cloud Platform free credits.
