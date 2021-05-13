This repository contains the code and experiments contained in the paper *"Analyzing Learning-Based Networked Systems with Formal Verification"* by Arnaud Dethise, Marco Canini and Nina Narodytska, published in INFOCOM 2021.

# Requirements

To run the experiments, the following is required:

- Python 3.7.9 (other versions were not tested)
- Tensorflow 2
- IBM ILOG CPLEX with Python bindings
- Install pypolyhedron and matplotlib (required for graphical output)

# Repository content

- `experiments.ipynb`: Contains the code used to encode properties and run experiments.
- `building_blocks.ipynb`: Contains the MILP implementation of the primitives presented in the paper.
- `pensieve.py`: Contains the MILP encoding of the Pensieve agent.
- `layers.py`: Contains generic implementation for Fully-Connected Linear and ReLU layers.
- `ilp/`, `utils/` and `ilp_utils*.py` contain support code with a high-level interface to interface with Cplex.
- `model/` contains a trained model of Pensieve which is used to extract the model parameters.
