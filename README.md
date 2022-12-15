# Neural Networks with Quantization Constraints.

  

Official Repo for the paper: [Neural networks with quantization constraints](https://arxiv.org/abs/2210.15623). Hounie, Elenter and Ribeiro.

Submitted to ICASSP 2023.

Our code is based on the [Any Precision DNNs repo](https://github.com/SHI-Labs/Any-Precision-DNNs).

*Repo under construction*

# Requirements

  

PyTorch>=1.1.0 and torchvision>=0.2.1

  

Other requirements can be installed via pip by running

```
pip install -r requirements.txt
```
  

# Train a model using PD-QAT


To train a model using PD-QAT on CIFAR-10 just run
```
python train_pd_layers.py
```
For additional information on available options and parameters run
```
python train_pd_layers.py --help
```

# Run Paper Experiments
Scripts used to generate the paper's result  can be found under the folder `scripts`.

## Baseline
To run QAT using DoReFa (baseline) for a given quantization bitwidth just run
```
./scripts/baseline_vanilla.sh {BITWIDTH}
```
Or to simply run baselines for Bitwidths 2,4,8 as reported in the paper:
```
./scripts/run_all_baselines.sh
```
## PD-QAT

### Output Constraints
To run QAT for a given quantization bidthwidth with a proximity constraint only at the models’ outputs run
```
./scripts/ours_only_CE.sh {BITWIDTH} {CONSTRAINT LEVEL}
```

### Layerwise Constraints

To run QAT for a given quantization bidthwidth with a proximity constraint only at the models’ outputs run
```
./scripts/ours_only_CE.sh {BITWIDTH} {CONSTRAINT LEVEL}
```
# Mixed Precision

To train a model implementing some layers in high precision run
```
python train_selec.py --no_quant_layer "{COMMA SEPARATED LAYER LIST}" --bit_width_list "{BITWIDTH}"
```
For example to implement a model in two bit precision with layers 6, 7 and 8 in high precision (BW=32)
```
python train_selec.py --no_quant_layer "6,7,8" --bit_width_list 32
```
Scripts used to train mixed precision models presented in the paper can be found in the folder `scripts/layer_selections `.

# Online Logging

All of our experiments were logged online using [*weights and biases*](https://wandb.ai/). In order to log your experiments add the argument --wandb_log (and sign in with your credentials) and customize the entity and project name under which results are logged.

# Notebooks

We also include Jupyter notebooks that pull results logged to weights and biases and create the plots and tables included in the paper.
Under the folder `notebooks`, three files are included:

*  `final_table.ipynb`: Outputs (in latex format) Table 1, a performance comparison of our method with respect to DoReFa.
* `per_layer_l2.ipynb`: Layer-wise constraints results. Generates Fig. 1.a (MSE between the high and low precision activations at different layers) and also sorts layers according to the final value of dual variables, which was used as an input for the layer selection experiments.
* `layer_select_results.ipynb`:  Generates Fig. 1.b and c showing the performance of mixed precision models when selecting layers leveraging dual variables.

