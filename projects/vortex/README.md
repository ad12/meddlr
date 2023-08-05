# Vortex in Meddlr
**Physics-Driven Data Augmentations for Consistency Training for Robust Accelerated MRI Reconstruction**

Arjun D Desai\*, Beliz Gunel\*, Batu M Ozturkler, Harris Beg, Shreyas Vasanawala, Brian A Hargreaves, Christopher Ré, John M Pauly, and Akshay S Chaudhari.

[arXiv](https://arxiv.org/abs/2111.02549) | [BibTeX](#citation)

<div align="center">
    <img src="https://drive.google.com/uc?export=view&id=1q0jAm6Kg5ZhRg3h0w0ZbtIgcRF3_-Vgb" alt="Vortex Schematic" width="700px" />
</div>

This project implements VORTEX in Meddlr. This is the official project hub for VORTEX.

## Usage
All training and evaluation can be done using [tools/train_net.py](../../tools/train_net.py)
and [tools/eval_net.py](../../tools/eval_net.py). Both are similar to the default methods
described in Meddlr's [GETTING_STARTED](../../GETTING_STARTED.md) guide.

### Training
To train VORTEX, you need to

```bash
python /path/to/meddlr/tools/train_net.py --config-file </path/to/config.yaml>
```

If you are using a config file in the [model zoo](MODEL_ZOO.md) and want to train on a GPU, you will need to override the `MODEL.DEVICE` argument:

```bash
python /path/to/meddlr/tools/train_net.py --config-file download://<cfg-url> MODEL.DEVICE cuda
```

### Evaluation
You can evaluate VORTEX with different combinations of physics-driven test-time perturbations (e.g. noise, motion).

```bash
# E.g. Standard evaluation on in-distribution (no perturbation) data with validation psnr checkpoint.
python /path/to/meddlr/tools/eval_net.py --config-file </path/to/config.yaml> --metric psnr_scan

# Add test-time motion perturbations of strength alpha=0.1,0.2,...,0.5
python /path/to/meddlr/tools/eval_net.py --config-file </path/to/config.yaml> --metric psnr_scan --motion sweep --motion-sweep-vals 0.1 0.2 0.3 0.4 0.5
```

### Model Zoo
Models and experiment configs are available in the [model zoo](MODEL_ZOO.md).

## Demo
Check out the interactive demo in `demo.ipynb` to see how VORTEX works in practice.

<div align="center">

<image src="static/vortex.gif" height=400 alt="MRI reconstruction under noise and motion perturbation" />
</div>


## Citation
If you use VORTEX in your work, please use the following BibTeX entry,
```
@article{desai2021vortex,
  title={VORTEX: Physics-Driven Data Augmentations for Consistency Training for Robust Accelerated MRI Reconstruction},
  author={Desai, Arjun D and Gunel, Beliz and Ozturkler, Batu M and Beg, Harris and Vasanawala, Shreyas and Hargreaves, Brian A and R{\'e}, Christopher and Pauly, John M and Chaudhari, Akshay S},
  journal={arXiv preprint arXiv:2111.02549},
  year={2021}
}
```