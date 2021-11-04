# Vortex in Meddlr
**Physics-Driven Data Augmentations for Consistency Training for Robust Accelerated MRI Reconstruction**

Arjun D Desai\*, Beliz Gunel\*, Batu M Ozturkler, Harris Beg, Shreyas Vasanawala, Brian A Hargreaves, Christopher Ré, John M Pauly, and Akshay S Chaudhari.

<div align="center">
    <img src="https://drive.google.com/uc?export=view&id=11ESUcZzfy4x4YGiBNqhNAXhqn4y9RWU-" alt="Vortex Schematic" width="700px" />
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

### Evaluation
You can evaluate VORTEX with different combinations of physics-driven test-time perturbations (e.g. noise, motion).

```bash
# E.g. Standard evaluation on in-distribution (no perturbation) data with validation psnr checkpoint.
python /path/to/meddlr/tools/eval_net.py --config-file </path/to/config.yaml> --metric psnr_scan

# Add test-time motion perturbations of strength alpha=0.1,0.2,...,0.5
python /path/to/meddlr/tools/eval_net.py --config-file </path/to/config.yaml> --metric psnr_scan --motion sweep --motion-sweep-vals 0.1 0.2 0.3 0.4 0.5
```
