# Noise2Recon in Meddlr
**A Semi-Supervised Framework for Joint MRI Reconstruction and Denoising**

Arjun D Desai\*, Batu M Ozturkler\*, Christopher M Sandino, Shreyas Vasanawala, Brian A Hargreaves, Christopher Ré, John M Pauly, and Akshay S Chaudhari.

[arXiv](https://arxiv.org/abs/2110.00075) | [BibTeX](#citation)

<div align="center">
    <img src="https://drive.google.com/uc?export=view&id=1k9W7Swcfms1d38Qfyet7422NGcQAyu3Q" alt="Noise2Recon Schematic" width="700px" />
</div>

Noise2Recon is a semi-supervised framework for joint MRI reconstruction and denoising. It uses a consistency branch to complement supervised MRI reconstruction with an unsupervised denoising objective. It also proposes an alternating/balanced sampling approach to stabilize optimization between supervised and unsupervised data in cases of large data imbalance.

## Usage
We provide template configs used for experiments in the main paper. 
All training and evaluation can be done using [tools/train_net.py](../../tools/train_net.py)
and [tools/eval_net.py](../../tools/eval_net.py). Both are similar to the default methods
described in Meddlr's [GETTING_STARTED](../../GETTING_STARTED.md) guide.

### Training
To train Noise2Recon, run

```bash
python /path/to/meddlr/tools/train_net.py --config-file </path/to/config.yaml>
```

If you are using a config file in the [model zoo](MODEL_ZOO.md) and want to train on a GPU, you will need to override the `MODEL.DEVICE` argument:

```bash
python /path/to/meddlr/tools/train_net.py --config-file download://<cfg-url> MODEL.DEVICE cuda
```

### Evaluation
You can evaluate Noise2Recon with physics-driven test-time noise perturbations.

```bash
# E.g. Standard evaluation on in-distribution (no perturbation) data with validation psnr checkpoint.
python /path/to/meddlr/tools/eval_net.py --config-file </path/to/config.yaml> --metric psnr_scan

# Add test-time motion perturbations of strength sigma=0.1,0.2,...,0.5
python /path/to/meddlr/tools/eval_net.py --config-file </path/to/config.yaml> --metric psnr_scan --noise sweep --sweep-vals 0.1 0.2 0.3 0.4 0.5
```

### Model Zoo
Models and experiment configs are available in the [model zoo](MODEL_ZOO.md).

## Citation
If you use Noise2Recon in your work, please use the following BibTeX entry,
```
@article{desai2021noise2recon,
  title={Noise2Recon: A Semi-Supervised Framework for Joint MRI Reconstruction and Denoising},
  author={Desai, Arjun D and Ozturkler, Batu M and Sandino, Christopher M and Vasanawala, Shreyas and Hargreaves, Brian A and Re, Christopher M and Pauly, John M and Chaudhari, Akshay S},
  journal={arXiv preprint arXiv:2110.00075},
  year={2021}
}
```