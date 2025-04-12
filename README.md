An implementation of Multivariate Adaptive Anomaly Detector (MAAD) using a Transformer-LSTM neural network architecture. 

The model aims to detect anomalous measurements from real-time multivariate time series data.
Utilizes multivariate prediction based on past measurements from different sensor nodes with each sensor node having multiple sensor parameters. 

This is an extension of AdapAD, a univariate and concept-drift-adaptive anomaly detector.

## References
> Nguyen, N.T., Heldal, R. and Pelliccione, P., 2024. Concept-drift-adaptive anomaly detector for marine sensor data streams. Internet of Things, p.101414.

```bibtex
@article{nguyen2024concept,
  title={Concept-drift-adaptive anomaly detector for marine sensor data streams},
  author={Nguyen, Ngoc-Thanh and Heldal, Rogardt and Pelliccione, Patrizio},
  journal={Internet of Things},
  pages={101414},
  year={2024},
  publisher={Elsevier}
}
```
https://github.com/ntnguyen-so/AdapAD_alg

## Installation

To install and use AdapAD from source, you will need the following tools:

- `git`
- `conda` (anaconda or miniconda)

#### Steps for installation

**Step 1:** Clone this repository using `git` and change into its root directory.

```bash
git clone https://github.com/vindenez/AdapAD_multivariate
cd AdapAD_multivariate
```

**Step 2:** Install necessary modules for `AdapAD_multivariate`.

```bash
pip install -r requirements.txt
```

**Step 3:** Installation complete!

## Usage

Execute the following command to run the algorithm
```bash
python3 main.py
```

You can find all the hyperparameters setting in `config.py`

## Anomaly Detection Example
![ATT-LSTM-0 0070](https://github.com/user-attachments/assets/4133eed8-e68e-457d-badf-3996c548be7d)
