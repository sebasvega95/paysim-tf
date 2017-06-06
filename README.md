# Fraud detection in synthetic financial dataset

Classification of fraudulent transactions using a synthetic dataset generated using the simulator called PaySim.

## Dataset

[Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/ntnu-testimon/paysim1)

It is assumed that you unzipped the file and you renamed it to `data.csv`.

## Setting

The code is designed to be run in a [Slurm](https://slurm.schedmd.com/)-managed cluster, where (hopefully) each node has a CUDA GPU.

## Dependencies

- [imbalanced-learn](http://contrib.scikit-learn.org/imbalanced-learn/)
- [numpy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [python-hostlist](https://www.nsc.liu.se/~kent/python-hostlist/)
- [PyZMQ](https://github.com/zeromq/pyzmq)
- [scikit-learn](http://scikit-learn.org/stable/)
- [TensorFlow](https://www.tensorflow.org/)

## Usage

```bash
  sbatch run-paysim-tf.sh
```
