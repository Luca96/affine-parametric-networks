"""CMS/plot"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import mplhep

from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

from script import utils
from script.cms.data import sample


def set_style(style=mplhep.style.LHCb2, dpi=100, **kwargs):
    """Sets the default style for plots.
        https://matplotlib.org/stable/tutorials/introductory/customizing.html#the-default-matplotlibrc-file
    """
    # reset old style
    mplhep.style.use(None)
    
    # use style from experiment
    mplhep.style.use(style)
    
    # further customization
    mpl.rcParams['lines.linewidth'] = 2.5
    mpl.rcParams['lines.markersize'] = 10
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['axes.axisbelow'] = True
    mpl.rcParams['grid.linestyle'] = 'dashed'
    mpl.rcParams['grid.alpha'] = 0.65
    mpl.rcParams['legend.frameon'] = True
    mpl.rcParams['legend.framealpha'] = 0.7
    mpl.rcParams['figure.figsize'] = (12, 10)
    mpl.rcParams['figure.autolayout'] = False
    mpl.rcParams['figure.dpi'] = dpi
    
    for k, v in kwargs.items():
        mpl.rcParams[k] = v


def roc_auc(true, pred, weights, cut: float, eps=1e-4, **kwargs):
    fpr, tpr, t = roc_curve(true, pred, sample_weight=weights)
    auc = roc_auc_score(true, pred, average='micro', sample_weight=weights)
    
    # find significance along the curve
    idx = (np.abs(cut - np.array(t))).argmin()

    return fpr, tpr, auc, fpr[idx], tpr[idx]


def pr_auc(true, pred, weights, cut: float, eps=1e-4, **kwargs):
    precision, recall, t = precision_recall_curve(true, pred, sample_weight=weights)
    auc = average_precision_score(true, pred, average='micro', sample_weight=weights)
    
    # find significance along the curve
    idx = (np.abs(cut - t)).argmin()
    
    return precision, recall, auc, precision[idx], recall[idx]
