[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6453048.svg)](https://doi.org/10.5281/zenodo.6453048)

# Tutorial: Affine Parametric Neural Networks

**Parametric Neural Networks** (pNNs) are a kind of neural networks, developed by [Baldi et al.](https://arxiv.org/pdf/1601.07913), which are mainly used for *signal-background classification* in High-Energy Physics (HEP). In our recent [paper](https://iopscience.iop.org/article/10.1088/2632-2153/ac917c), we propose various improvements to the original pNN, 
such as: 
* The *affine architecture*: based on interleaving multiple **affine-conditioning layers**;
* Guidelines on how to assign the *physics parameter* (e.g. the particle mass);
* The *balanced training* procedure, in which we build balanced mini-batches by leveraging the structure of both the 
signal and background.

This branch of the repo is about a **complete tutorial** on how to define, and apply pNNs: everything is described in 
the `tutorial.ipynb` notebook.

---
## Installation and Usage

0. Open a terminal: make sure to have both Python and Jupyter notebook (or lab) installed.
1. Clone the repository (but only the `tutorial` branch):

   ```bash
   git clone https://github.com/Luca96/affine-parametric-networks.git --branch tutorial
   # if on Google Colab, use %cd
   cd affine-parametric-networks
   ```

2. Just run the `tutorial.ipynb` notebook; that's it.

---

## How to Cite

If you use the code and/or the dataset we provide for your own project or research, please cite our paper:

```latex
@article{Anzalone_2022,
   doi = {10.1088/2632-2153/ac917c},
   url = {https://doi.org/10.1088/2632-2153/ac917c},
   year = 2022,
   month = {sep},
   publisher = {{IOP} Publishing},
   volume = {3},
   number = {3},
   pages = {035017},
   author = {Luca Anzalone and Tommaso Diotalevi and Daniele Bonacorsi},
   title = {Improving parametric neural networks for high-energy physics (and beyond)},
   journal = {Machine Learning: Science and Technology}
}
```

Dataset citation:

```
@dataset{hepmass_imb,
  author={Luca Anzalone and Tommaso Diotalevi and Daniele Bonacorsi},
  title={HEPMASS-IMB},
  month=apr,
  year=2022,
  publisher={Zenodo},
  doi={10.5281/zenodo.6453048},
  url={https://doi.org/10.5281/zenodo.6453048}
}
```

