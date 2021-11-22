"""CMS/plot"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from script import utils
from script.datasets import Dataset
from script.cms.data import sample

from typing import Union


def significance(model, dataset: Dataset, mass: int, category: int, signal: str, delta=50, bins=20, 
         size=(12, 10), legend='best', name='Model', seed=utils.SEED,
         path='plot', save=None, show=True, ax=None):
    """Plots the output distribution of the model, along it's weighted significance"""
    
    if ax is None:
        fig = plt.figure(figsize=size)
        ax = fig.gca()
    
    # select both signal and background in interval (m-d, m+d)
    sig = dataset.signal[dataset.signal['mA'] == mass]
    sig = sig[(sig['dimuon_M'] >= mass - delta) & (sig['dimuon_M'] < mass + delta)]

    bkg = dataset.background
    bkg = bkg[(bkg['dimuon_M'] >= mass - delta) & (bkg['dimuon_M'] < mass + delta)]
    
    num_sig = sig.shape[0]
    
    # prepare data
    x = pd.concat([sig[dataset.columns['feature']],
                   bkg[dataset.columns['feature']]], axis=0).values

    y = np.reshape(pd.concat([sig['type'], bkg['type']], axis=0).values, newshape=[-1])
    x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
    # predict data
    out = model.predict(x=x, batch_size=1024, verbose=0)
    out = np.asarray(out)

    y_sig = np.squeeze(out[y == 1.0])
    y_bkg = np.squeeze(out[y == 0.0])

    w_bkg = bkg['weight'].values
    w_sig = np.ones_like(y_sig)
    
    # plot
    names = dataset.names_df.loc[bkg.index.values]
    df = pd.DataFrame({'Output': y_bkg, 'Bkg': np.squeeze(names), 'weight': w_bkg})
    
    # plot histograms
    sns.histplot(data=df, x='Output', hue='Bkg', multiple='stack', edgecolor='.3', linewidth=0.5, bins=bins,
                 weights='weight', ax=ax, binrange=(0.0, 1.0),
                 palette={'DY': 'green', 'TTbar': 'red', 'ST': 'blue', 'diboson': 'yellow'})
    
    h_bkg, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)
    h_bkg = np.sum(h_bkg)
    
    h_sig, _ = np.histogram(y_sig, bins=bins)
    h_sig = np.sum(h_sig)
    
    w_sig = np.ones_like(y_sig) * (h_bkg / h_sig)
    
    ax.hist(y_sig, bins=bins, alpha=0.5, label='signal', color='purple', edgecolor='purple', 
            linewidth=2, hatch='//', histtype='step', range=(0.0, 1.0),
            weights=w_sig)
    
    # compute significance
    sig_mask = np.squeeze(y == 1.0)
    bkg_mask = np.squeeze(y == 0.0)

    cuts = np.linspace(0.0, 1.0, num=bins)
    ams = []
    w = np.concatenate([w_sig, w_bkg], axis=0)
    
    bx = ax.twinx()
    
    s, _ = np.histogram(y_sig, bins=bins, weights=w_sig)
    b, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)
    
    for i in range(s.shape[0]):
        s_i = np.sum(s[i:])
        b_i = np.sum(b[i:])
        
        ams.append(s_i / np.sqrt(s_i + b_i))

    k = np.argmax(ams)
    
    # add stuff to plot
    bx.grid(False)
    bx.plot(cuts, ams, color='g', label='Significance')

    ax.axvline(x=cuts[k], linestyle='--', linewidth=2, color='g',
               label=f'{round(cuts[k], 3)}: {round(ams[k], 3)}')

    bx.set_ylabel(r'Significance: $s/\sqrt{s+b}$')
    
    leg = ax.get_legend()
    ax.legend(loc='upper left')
    ax.add_artist(leg)
    
    ax.set_xlabel('Class Label Probability')
    ax.set_ylabel('Weighted Num. Events')
    
    # title
    str0 = f'Category-{category} (#bins = {bins})'
    str1 = f'@{(int(mass - delta), int(mass + delta))} dimuon_M (bkg)'
    str2 = f'{name} Output Distribution @ {int(mass)}mA (signal {signal}), {str1}'
    str3 = f'# signal = {sig.shape[0]}, # bkg = {bkg.shape[0]}'
    
    ax.set_title(f'{str0}\n{str2}\n{str3}')
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}.png'), bbox_inches='tight')
    
    if show:
        plt.show()


def compare_significance(models: list, dataset: Dataset, mass: float, category: int, *args, path='plot', save=None, 
                         size=(12, 10), share_y=True, **kwargs):
    fig, axes = plt.subplots(nrows=1, ncols=len(models), sharey=bool(share_y))
    
    fig.set_figwidth(size[0] * len(models))
    fig.set_figheight(size[1])
    
    for i, model in enumerate(models):
        significance(model, dataset, mass, category, *args, **kwargs, save=None, show=False, ax=axes[i])
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}_{int(mass)}mA.png'), bbox_inches='tight')
    
    plt.show()


def cut(model, dataset: Dataset, category: int, signal: str, delta=50, bins=20, 
        size=(12, 10), legend='best', name='Model', seed=utils.SEED,
        path='plot', save=None, show=True, ax=None):
    """Plots the value of the best cut as the mass (mA) varies"""
    if ax is None:
        fig = plt.figure(figsize=size)
        ax = fig.gca()
    
    signal = dataset.signal
    backgr = dataset.background
    
    cuts = []
    
    # select both signal and background in interval (m-d, m+d)
    for mass in dataset.unique_signal_mass:
        sig = signal[signal['mA'] == mass]
        sig = sig[(sig['dimuon_M'] >= mass - delta) & (sig['dimuon_M'] < mass + delta)]

        bkg = backgr
        bkg = bkg[(bkg['dimuon_M'] >= mass - delta) & (bkg['dimuon_M'] < mass + delta)]

        # prepare data
        x = pd.concat([sig[dataset.columns['feature']],
                       bkg[dataset.columns['feature']]], axis=0).values

        y = np.reshape(pd.concat([sig['type'], bkg['type']], axis=0).values, newshape=[-1])
        x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)

        # predict data
        out = model.predict(x=x, batch_size=1024, verbose=0)
        out = np.asarray(out)

        y_sig = np.squeeze(out[y == 1.0])
        y_bkg = np.squeeze(out[y == 0.0])
    
        # computing weights
        w_bkg = bkg['weight'].values
        w_sig = np.ones_like(y_sig)

        h_bkg, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)
        h_bkg = np.sum(h_bkg)

        h_sig, _ = np.histogram(y_sig, bins=bins)
        h_sig = np.sum(h_sig)

        w_sig = np.ones_like(y_sig) * (h_bkg / h_sig)

        # compute significance
        ams = []

        s, _ = np.histogram(y_sig, bins=bins, weights=w_sig)
        b, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)

        for i in range(s.shape[0]):
            s_i = np.sum(s[i:])
            b_i = np.sum(b[i:])

            ams.append(s_i / np.sqrt(s_i + b_i))

        k = np.argmax(ams)
        cuts.append(np.linspace(0.0, 1.0, num=bins)[k])  # add cut value
    
    # plot
    ax.plot(dataset.unique_signal_mass, cuts, marker='o', label=f'avg {round(np.mean(cuts), 2)}')
    
    ax.set_xlabel('Mass (GeV)')
    ax.set_ylabel('Best Cut')
    
    ax.legend(loc=str(legend))
    
    # title
    str0 = f'Category-{category} [#bins = {bins}]'
    ax.set_title(f'[{name}] Best Cut vs mA ({str0})')
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}.png'), bbox_inches='tight')
    
    if show:
        plt.show()
    
    return cuts


def curves(model, dataset: Dataset, mass: int, category: int, signal: str, delta=50, 
           bins=20, size=(10, 9), legend='best', seed=utils.SEED, path='plot', save=None):
    """Plots the PR and ROC curves"""
    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    fig.set_figwidth(size[0] * 2)
    fig.set_figheight(size[1])
    
    sig = dataset.signal[dataset.signal['mA'] == mass]
    sig = sig[(sig['dimuon_M'] >= mass - delta) & (sig['dimuon_M'] < mass + delta)]
    
    bkg = dataset.background
    bkg = bkg[(bkg['dimuon_M'] >= mass - delta) & (bkg['dimuon_M'] < mass + delta)]
    
    num_sig = sig.shape[0]
    
    # prepare data
    x = pd.concat([sig[dataset.columns['feature']],
                   bkg[dataset.columns['feature']]], axis=0).values

    y = np.reshape(pd.concat([sig['type'], bkg['type']], axis=0).values, newshape=[-1])
    x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
    # predict data
    out = model.predict(x=x, batch_size=1024, verbose=0)
    out = np.asarray(out)
    
    y_sig = np.squeeze(out[y == 1.0])
    y_bkg = np.squeeze(out[y == 0.0])
    
    # compute weights
    w_bkg = bkg['weight'].values
    
    h_bkg, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)
    h_bkg = np.sum(h_bkg)
    
    h_sig, _ = np.histogram(y_sig, bins=bins)
    h_sig = np.sum(h_sig)
    
    w_sig = np.ones_like(y_sig) * (h_bkg / h_sig)
    w = np.concatenate([w_sig, w_bkg], axis=0)
    
    str1 = f'@{(int(mass - delta), int(mass + delta))} dimuon_M (bkg)'
    
    # PR-curve
    PrecisionRecallDisplay.from_predictions(y_true=y, y_pred=out, sample_weight=w, ax=axes[0],
                                            name=f'pNN @ {int(mass)}mA (signal {signal}), {str1}')
    axes[0].set_title(f'[pNN] PR Curve @ {int(mass)}mA (category {category})')
    
    # ROC curve
    RocCurveDisplay.from_predictions(y_true=y, y_pred=out, sample_weight=w, ax=axes[1],
                                     name=f'pNN @ {int(mass)}mA (signal {signal}), {str1}')
    axes[1].set_title(f'[pNN] ROC Curve @ {int(mass)}mA (category {category})')
    
    fig.tight_layout()
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}.png'), bbox_inches='tight')
    
    plt.show()


def roc_auc(true, pred, weights, cut: float, eps=1e-4):
    fpr, tpr, t = roc_curve(true, pred, sample_weight=weights)
    auc = roc_auc_score(true, pred, average='micro', sample_weight=weights)
    
    # find significance along the curve
    idx = (np.abs(cut - np.array(t))).argmin()

    return fpr, tpr, auc, fpr[idx], tpr[idx]


def pr_auc(true, pred, weights, cut: float, eps=1e-4):
    precision, recall, t = precision_recall_curve(true, pred, sample_weight=weights)
    auc = average_precision_score(true, pred, average='micro', sample_weight=weights)
    
    # find significance along the curve
    idx = (np.abs(cut - t)).argmin()
    
    return precision, recall, auc, precision[idx], recall[idx]


def compare_roc(dataset: Dataset, models_and_cuts: dict, mass: float, category: int, delta=50.0, bins=20, 
		size=(12, 10), digits=3, path='plot', save=None, name='Model', **kwargs):
    
    def get_predictions_and_weights(model, x, y, bkg):
        out = model.predict(x=x, batch_size=1024, verbose=0)
        out = np.asarray(out)

        y_sig = np.squeeze(out[y == 1.0])
        y_bkg = np.squeeze(out[y == 0.0])

        # compute weights
        w_bkg = bkg['weight'].values

        h_bkg, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)
        h_bkg = np.sum(h_bkg)

        h_sig, _ = np.histogram(y_sig, bins=bins)
        h_sig = np.sum(h_sig)

        w_sig = np.ones_like(y_sig) * (h_bkg / h_sig)
        w = np.concatenate([w_sig, w_bkg], axis=0)
        
        return out, w
    
    sig = dataset.signal[dataset.signal['mA'] == mass]
    sig = sig[(sig['dimuon_M'] >= mass - delta) & (sig['dimuon_M'] < mass + delta)]
    
    bkg = dataset.background
    bkg = bkg[(bkg['dimuon_M'] >= mass - delta) & (bkg['dimuon_M'] < mass + delta)]

    # prepare data
    x = pd.concat([sig[dataset.columns['feature']],
                   bkg[dataset.columns['feature']]], axis=0).values

    y = np.reshape(pd.concat([sig['type'], bkg['type']], axis=0).values, newshape=[-1])
    x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
    # predict data
    plt.figure(figsize=size)
    plt.title(f'[{name}] ROC @ {int(mass)}mA [category-{int(category)}]')
    
    for k, (model, cut) in models_and_cuts.items():
        out, w = get_predictions_and_weights(model, x, y, bkg)
        
        fpr, tpr, auc, cut_fpr, cut_tpr = roc_auc(true=y, pred=out, weights=w, cut=cut, **kwargs)
    
        plt.plot(fpr, tpr, label=f'AUC ({k}) {np.round(auc, digits)}')
        plt.scatter(cut_fpr, cut_tpr, label=f'significance @ {round(cut, digits)}')
    
    plt.xlabel('Background Efficiency (False Positive Rate)')
    plt.ylabel('Signal Efficienty (True Positive Rate)')
    
    plt.legend(loc='lower right')
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}_{int(mass)}mA.png'), bbox_inches='tight')
    
    plt.show()

    
def compare_pr(dataset: Dataset, models_and_cuts: dict, category: int, mass: float, delta=50.0, bins=20, 
	   size=(12, 10), digits=3, path='plot', save=None, name='Model', **kwargs):

    def get_predictions_and_weights(model, x, y, bkg):
        out = model.predict(x=x, batch_size=1024, verbose=0)
        out = np.asarray(out)

        y_sig = np.squeeze(out[y == 1.0])
        y_bkg = np.squeeze(out[y == 0.0])

        # compute weights
        w_bkg = bkg['weight'].values

        h_bkg, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)
        h_bkg = np.sum(h_bkg)

        h_sig, _ = np.histogram(y_sig, bins=bins)
        h_sig = np.sum(h_sig)

        w_sig = np.ones_like(y_sig) * (h_bkg / h_sig)
        w = np.concatenate([w_sig, w_bkg], axis=0)
        
        return out, w
    
    sig = dataset.signal[dataset.signal['mA'] == mass]
    sig = sig[(sig['dimuon_M'] >= mass - delta) & (sig['dimuon_M'] < mass + delta)]
    
    bkg = dataset.background
    bkg = bkg[(bkg['dimuon_M'] >= mass - delta) & (bkg['dimuon_M'] < mass + delta)]

    # prepare data
    x = pd.concat([sig[dataset.columns['feature']],
                   bkg[dataset.columns['feature']]], axis=0).values

    y = np.reshape(pd.concat([sig['type'], bkg['type']], axis=0).values, newshape=[-1])
    x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
    # predict data
    plt.figure(figsize=size)
    plt.title(f'[{name}] PR Curve @ {int(mass)}mA [category-{int(category)}]')
    
    for k, (model, cut) in models_and_cuts.items():
        out, w = get_predictions_and_weights(model, x, y, bkg)
        
        precision, recall, auc, cut_prec, cut_rec = pr_auc(true=y, pred=out, weights=w, cut=cut, **kwargs)
        
        plt.plot(recall, precision, label=f'AUC ({k}) {np.round(auc, digits)}')
        plt.scatter(cut_rec, cut_prec, label=f'significance @ {round(cut, digits)}')
    
    plt.xlabel('Recall (signal efficiency)')
    plt.ylabel('Precision (purity)')
    
    plt.legend(loc='lower left')
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}_{int(mass)}mA.png'), bbox_inches='tight')
    
    plt.show()


def var_posteriori(dataset: Dataset, models: list, variables: list, mass: float, cuts: list, category: int, case: int, 
                   delta=50.0, size=(12, 10), legend='best', bins=25, share_y=True,
                   path='plot', weight=False, seed=utils.SEED, save=None, min_limit=None, max_limit=None):
    assert len(models) == 2, 'only two models can be compared'

    def predict(model, x, cut):
        out = model.predict(x=x, batch_size=1024, verbose=0)
        out = np.asarray(out)
    
        sig_mask = np.squeeze(out >= cut) & (y == 1.0)
        bkg_mask = np.squeeze(out < cut) & (y == 0.0)

        y_sig = np.squeeze(out[sig_mask])
        y_bkg = np.squeeze(out[bkg_mask])

        w_bkg = ds[bkg_mask]['weight'].values
        w_sig = np.ones_like(y_sig)

        # compute signal weights
        h_bkg, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)
        h_bkg = np.sum(h_bkg)

        h_sig, _ = np.histogram(y_sig, bins=bins)
        h_sig = np.sum(h_sig)

        w_sig = np.ones_like(y_sig) * (h_bkg / h_sig)
        names = dataset.names_df.loc[ds[bkg_mask].index]
        
        return out, (sig_mask, bkg_mask), (y_sig, y_bkg), (w_sig, w_bkg), names 
    
    sig = dataset.signal[dataset.signal['mA'] == mass]
    bkg = dataset.background
    
    # case
    if case == 2:
        sig = sig[(sig['dimuon_M'] >= mass - delta) & (sig['dimuon_M'] < mass + delta)]
        bkg = bkg[(bkg['dimuon_M'] >= mass - delta) & (bkg['dimuon_M'] < mass + delta)]
    
    ds = pd.concat([sig, bkg], axis=0)
    num_sig = sig.shape[0]
    
    # prepare data
    x = pd.concat([sig[dataset.columns['feature']],
                   bkg[dataset.columns['feature']]], axis=0).values

    y = np.reshape(pd.concat([sig['type'], bkg['type']], axis=0).values, newshape=[-1])
    x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
    # predict data
    out1, masks1, y1, w1, names1 = predict(models[0], x, cuts[0])
    out2, masks2, y2, w2, names2 = predict(models[1], x, cuts[1])
    
    out = [out1, out2]
    mask = [masks1, masks2]
    labels = [y1, y2]
    w = [w1, w2]
    names = [names1, names2]
    
    for col in variables:
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=bool(share_y))
        
        fig.set_figwidth(size[0] * 2)
        fig.set_figheight(size[1])
        
        ax = fig.gca()
        
        for i, ax in enumerate(axes):
            s = ds[mask[i][0]][col]
            b = ds[mask[i][1]][col]
        
            df = pd.DataFrame({col: b, 'Bkg': np.squeeze(names[i]), 'weight': w[i][1]})

            if ('met' in col) or ('pt' in col):
                n_bins = 50
                max_limit = 500.0

            elif col == 'ljet_n':
                n_bins = 25
                max_limit = 11.0

            elif col == 'dimuon_M':
                n_bins = 25
                max_limit = 500.0
                min_limit = 300
            else:
                n_bins = 25
                max_limit = None

            range_min = min(b.min(), s.min())
            range_max = max(b.max(), s.max())

            if isinstance(min_limit, (int, float)):
                range_min = max(range_min, min_limit)

            if isinstance(max_limit, (int, float)):
                range_max = min(range_max, max_limit)

            # plot histograms
            sns.histplot(data=df, x=col, hue='Bkg', multiple='stack', edgecolor='.3', linewidth=0.5, bins=n_bins,
                         weights='weight', ax=ax, binrange=(range_min, range_max),
                         palette={'DY': 'green', 'TTbar': 'red', 'ST': 'blue', 'diboson': 'yellow'})

            ax.hist(s, bins=n_bins, alpha=0.7, label='signal', color='purple', edgecolor='purple', 
                    linewidth=2, hatch='//', histtype='step',
                    range=(range_min, range_max), weights=w[i][0])

            leg = ax.get_legend()
            ax.legend(loc='upper left')
            ax.add_artist(leg)

            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            
            fig.tight_layout()
        
        if isinstance(save, str):
            path = utils.makedir(path)
            plt.savefig(os.path.join(path, f'{save}_{col}_{int(mass)}_mA.png'), bbox_inches='tight')
        
        plt.show()
