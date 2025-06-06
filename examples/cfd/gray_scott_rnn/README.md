# RNN for transient 3D Gray Scott system

This example uses recurrent neural networks for spatio-temporal prediction of the
Gray-Scott reaction diffusion system. The example uses architecture that is inspired from
[Physics Informed RNN-DCT Networks for Time-Dependent Partial Differential Equations](https://arxiv.org/pdf/2202.12358.pdf)
paper.

## Problem overview

Time-series prediction is a key task in many domains.
The application of deep learning architectures—particularly RNNs, long short-term memory
networks (LSTMs), and similar networks has significantly enhanced the predictive capabilities.
These models are unique in their ability to capture temporal dependencies and learn complex
patterns over time, making them well suited for forecasting time varying relationships.
In physics-ML, these models are critical in predicting dynamic physical systems’ evolution,
enabling better simulations, understanding of complex natural phenomena, and aiding
in discoveries.

This problem involves predicting the next timesteps of a 3D Gray Scott system given the
initial condition.

## Dataset

This example relies on the Dataset used in [Transformers for modeling physical systems](https://www.sciencedirect.com/science/article/abs/pii/S0893608021004500)
paper which solves the Reaction-Diffusion system governed by Gray-Scott model on a 3D grid.
The different samples are generated by using different initial conditions for the simulation.

## Model overview and architecture

The model uses Convolutional GRU layers for the RNN propagation and use a ResNet type
architecture for spatial encoding. This example uses the one-to-many variant of the RNN
model in PhysicsNeMo.

![Comparison between the 3D RNN model prediction and the
ground truth](../../../docs/img/gray_scott_predictions_blog_2.gif)

## Prerequisites

Install the requirements using:

```bash
pip install -r requirements.txt
```

## Getting Started

The example script contains code to download and do any pre-processing for the dataset.

To get started, simply run

```bash
python gray_scott_rnn.py
```

## References

- [Physics Informed RNN-DCT Networks for Time-Dependent Partial Differential Equations](https://arxiv.org/pdf/2202.12358.pdf)
- [Transformers for modeling physical systems](https://www.sciencedirect.com/science/article/abs/pii/S0893608021004500)
