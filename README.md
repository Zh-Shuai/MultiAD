# MultiAD: Multiple Hypothesis Testing for Anomaly Detection in Multi-type Event Sequences
Pytorch implementation of the paper ["Multiple Hypothesis Testing for Anomaly Detection in Multi-type Event Sequences"], by Shuai Zhang, Chuan Zhou, Peng Zhang, Yang Liu, Zhao Li, and Hongyang Chen, ICDM 2023.


## Installation
1. Install the dependencies
    ```
    conda env create -f environment.yml
    ```
2. Activate the conda environment
    ```
    conda activate anomaly_mpp
    ```
3. Install the package (this command must be run in the `MultiAD` folder)
    ```
    pip install -e .
    ```
4. Unzip the data
    ```
    unzip data.zip
    ```

## Reproducing the results from the paper
- `experiments/spp.py`: GOF testing for the standard Poisson process (Section V-A in the paper).
- `experiments/multivariate.py`: Detecting anomalies in synthetic data (Section V-B).
- `experiments/real_world.py`: Detecting anomalies in real-world data (Section V-C).


## Citation
If you find this code useful, please consider citing our paper: Multiple Hypothesis Testing for Anomaly Detection in Multi-type Event Sequences.

```

```

## Acknowledgements and References
Parts of this code are based on and/or copied from the code of: https://github.com/shchur/tpp-anomaly-detection, of the paper ["Detecting Anomalous Event Sequences with Temporal Point Processes"](https://papers.neurips.cc/paper/2021/hash/6faa8040da20ef399b63a72d0e4ab575-Abstract.html).