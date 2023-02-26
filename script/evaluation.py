
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from script import utils
from script import hep


def compare_table(models: dict, dataset, bins=100, ratio=False, **kwargs):
    """Computes classification metrics (AUC of ROC and PR, and AMS) in a df format"""
    # ROC & PR
    roc = hep.plot.get_curve_auc(dataset, models, which='ROC', **kwargs)
    utils.free_mem()

    pr = hep.plot.get_curve_auc(dataset, models, which='PR', **kwargs)
    utils.free_mem()

    # significance
    ams = {}

    for k, model in models.items():
        v, _ = hep.plot.get_ams_and_cut(model, dataset, bins=bins, ratio=ratio)
        ams[k] = v
        utils.free_mem()

    return roc, pr, ams


def pivot(df: pd.DataFrame, dataset, digits=4):
    """Helps to visualize the provided dataframe"""
    df2 = pd.DataFrame(df.values.T)
    df2.rename(columns=dict(enumerate(dataset.unique_signal_mass)), inplace=True)

    df2['mean'] = np.mean(df.values, axis=0)
    df2['model'] = df.columns

    return df2.round(digits)


def compare(models: dict, dataset, bins=100, ratio=True, **kwargs):
    """Compares the models on each mass point"""
    # ROC & PR
    ax1, ax2 = utils.get_plot_axes(rows=1, cols=2)

    hep.plot.auc_vs_mass(dataset, models, ax=ax1, which='ROC', **kwargs)
    utils.free_mem()

    hep.plot.auc_vs_mass(dataset, models, ax=ax2, which='PR', **kwargs)
    utils.free_mem()

    plt.tight_layout()
    plt.show()

    # significance
    hep.significance_vs_mass(models, dataset, bins=bins, ratio=ratio, **kwargs)
    utils.free_mem()
