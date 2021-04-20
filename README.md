# Parametric Neural Network for Higgs to MuMu

Code for the signal/background discrimination using the Parametric neural network approach (mentioned here [1])

Content of this repository:

- data/: Folder containing the dataset used for this analysis (~1.8GB);
    * signal.csv -> entire signal dataset (with mA as mass parameter);
	* background.csv -> entire background dataset (mA distributed as a random uniform);
	* AllData.csv -> 2016 Data (not used for training of the pNN - for the moment you can delete it!)
	* signal/ -> Folder containing the csv file for each mass point;
	* background/ -> Folder containing the csv file for each background;
- CMS\_graphics/ -> Folder with the python module used for CMS graph aesthetics;
- saved\_models/ -> Folder with the hdf5 model weights (model.h5 for the mass as single feature, model2.h5 for the mass as linear combination);
- pNN.ipynb -> Main jupyter notebook containing the code for the parametric Neural Network (the problem is here!)
- pyReader.ipynb -> This jupyter is the "next step" when the network starts working, making the inference on real data;

[1] Parameterized neural networks for high-energy physics. Pierre Baldi, Kyle Cranmer et al. arXiv: 1601.07913