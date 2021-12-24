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


PALETTE = {'DY': 'green', 'TTbar': 'red', 'ST': 'blue', 'diboson': 'yellow', 
           'ZMM': 'purple', 'signal': 'black'}


def significance(model, dataset: Dataset, mass: int, title='', interval=50, digits=4,
                 bins=20, size=(12, 10), legend='best', name='Model', palette=PALETTE, signal_in_interval=False,
                 path='plot', save=None, show=True, ax=None, weight_column='weight', ratio=False):
    """Plots the output distribution of the model, along it's weighted significance"""
    if ax is None:
        fig = plt.figure(figsize=size)
        ax = fig.gca()
    
    if isinstance(interval, (int, float)):
        interval = (mass - interval, mass + interval)

    # select both signal and background in interval (m-d, m+d)
    sig = dataset.signal[dataset.signal['mA'] == mass]

    if signal_in_interval:
        sig = sig[(sig['dimuon_mass'] >= interval[0]) & (sig['dimuon_mass'] < interval[1])]

    bkg = dataset.background
    bkg = bkg[(bkg['dimuon_mass'] >= interval[0]) & (bkg['dimuon_mass'] < interval[1])]
    
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

    w_bkg = bkg[weight_column].values
    w_sig = sig[weight_column].values
    
    # plot
    names = dataset.names_df.loc[bkg.index.values]
    df = pd.DataFrame({'Output': y_bkg, 'Bkg': np.squeeze(names), 'weight': w_bkg})
    
    # plot histograms
    sns.histplot(data=df, x='Output', hue='Bkg', multiple='stack', edgecolor='.3', linewidth=0.5, bins=bins,
                 weights='weight', ax=ax, binrange=(0.0, 1.0),
                 palette=palette)
    
    h_bkg, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)
    h_bkg = np.sum(h_bkg)
    
    h_sig, _ = np.histogram(y_sig, bins=bins)
    h_sig = np.sum(h_sig)
    
    w_sig = w_sig * (h_bkg / h_sig)
    
    ax.hist(y_sig, bins=bins, alpha=0.5, label='signal', color=palette['signal'], 
            edgecolor=palette['signal'], linewidth=2, hatch='//', histtype='step', 
            range=(0.0, 1.0), weights=w_sig)
    
    # compute significance
    sig_mask = np.squeeze(y == 1.0)
    bkg_mask = np.squeeze(y == 0.0)

    cuts = np.linspace(0.0, 1.0, num=bins)
    ams = []
    # ams_max = (np.sum(sig_mask) * w_sig[0]) / np.sqrt(np.sum(sig_mask) * w_sig[0])
    ams_max = np.sum(w_sig) / np.sqrt(np.sum(w_sig))
    
    bx = ax.twinx()
    
    s, _ = np.histogram(y_sig, bins=bins, weights=w_sig)
    b, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)
    
    for i in range(s.shape[0]):
        s_i = np.sum(s[i:])
        b_i = np.sum(b[i:])
        
        ams.append(s_i / np.sqrt(s_i + b_i))
    
    if ratio:
        ams = np.array(ams) / ams_max
    
    k = np.argmax(ams)
    
    # add stuff to plot
    bx.grid(False)
    bx.plot(cuts, ams, color='g', label='Significance')

    ax.axvline(x=cuts[k], linestyle='--', linewidth=2, color='g',
               label=f'{round(cuts[k], digits)}: {round(ams[k].item(), digits)}')
    
    if ratio:
        bx.set_ylabel(r'Significance Ratio: $(s\cdot\sqrt{s_\max}) /(s_\max\cdot\sqrt{s+b})$')
    else:
        bx.set_ylabel(r'Significance: $s / \sqrt{s+b}$')
    
    leg = ax.get_legend()
    ax.legend(loc='upper left')
    ax.add_artist(leg)
    
    ax.set_xlabel('Class Label Probability')
    ax.set_ylabel('Weighted Num. Events')
    
    # title
    str0 = f'{title} (#bins = {bins})'
    str1 = f'@{interval} dimuon_mass (bkg)'
    str2 = f'{name} Output Distribution @ {int(mass)}mA, {str1}'
    str3 = f'# signal = {sig.shape[0]}, # bkg = {bkg.shape[0]}'
    
    ax.set_title(f'{str0}\n{str2}\n{str3}')
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}.png'), bbox_inches='tight')
    
    if show:
        plt.show()
    
    return np.max(ams), cuts[k]


def _compute_significance(model, dataset: Dataset, bins=20, weight_column='weight', signal_in_interval=False):
    ams = []
    cuts = np.linspace(0.0, 1.0, num=bins)
    
    for mass, interval in zip(dataset.unique_signal_mass, dataset.mass_intervals):
        # select both signal and background in interval (m-d, m+d)
        sig = dataset.signal[dataset.signal['mA'] == mass]
        
        if signal_in_interval:
            sig = sig[(sig['dimuon_mass'] >= interval[0]) & (sig['dimuon_mass'] < interval[1])]

        bkg = dataset.background
        bkg = bkg[(bkg['dimuon_mass'] > interval[0]) & (bkg['dimuon_mass'] < interval[1])]
    
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
        w_bkg = bkg[weight_column].values
        w_sig = sig[weight_column].values
    
        h_bkg, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)
        h_bkg = np.sum(h_bkg)

        h_sig, _ = np.histogram(y_sig, bins=bins)
        h_sig = np.sum(h_sig)

        w_sig = w_sig * (h_bkg / h_sig)    

        # compute significance
        sig_mask = np.squeeze(y == 1.0)
        bkg_mask = np.squeeze(y == 0.0)

        ams_ = []
        s, _ = np.histogram(y_sig, bins=bins, weights=w_sig)
        b, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)

        for i in range(s.shape[0]):
            s_i = np.sum(s[i:])
            b_i = np.sum(b[i:])

            ams_.append(s_i / np.sqrt(s_i + b_i))

        ams.append(ams_)
    
    return np.max(ams, axis=-1), cuts[np.argmax(ams, axis=-1)]


def compare_significance(models_and_data: dict, mass: float, *args, path='plot', save=None, 
                         size=(12, 10), share_y=True, **kwargs):
    fig, axes = plt.subplots(nrows=1, ncols=len(models_and_data), sharey=bool(share_y))
    
    fig.set_figwidth(size[0] * len(models_and_data))
    fig.set_figheight(size[1])
    
    for i, (key, (model, data)) in enumerate(models_and_data.items()):
        significance(model, data, mass, *args, **kwargs, name=key, save=None, show=False, ax=axes[i])
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}_{int(mass)}mA.png'), bbox_inches='tight')
    
    plt.show()

    
def significance_ratio_vs_mass(models_and_data: dict, title='', weight_column='weight', xticks=None,
                               bins=20, size=(10, 9), path='plot', save=None, signal_in_interval=False):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    fig.set_figwidth(size[0] * 2)
    fig.set_figheight(size[1])
    
    if isinstance(models_and_data, tuple):
        models_and_data = {'': models_and_data}
    
    ams = {}
    cut = {}
    
    for name, (model, dataset) in models_and_data.items():
        ams[name] = []
        cut[name] = []
        
        for mass, (m_low, m_up) in zip(dataset.unique_signal_mass, dataset.mass_intervals):
            sig = dataset.signal
            
            # select data
            s = sig[sig['mA'] == mass]

            if signal_in_interval:
                s = s[(s['dimuon_mass'] >= m_low) & (s['dimuon_mass'] < m_up)]

            b = dataset.background
            b = b[(b['dimuon_mass'] >= m_low) & (b['dimuon_mass'] < m_up)]

            # prepare data
            x = pd.concat([s[dataset.columns['feature']],
                           b[dataset.columns['feature']]], axis=0).values

            y = np.reshape(pd.concat([s['type'], b['type']], axis=0).values, newshape=[-1])
            x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
            # predict data
            out, y_pred, w = _get_predictions_and_weights(model, x, y, sig=s, bkg=b, bins=bins, weight_column=weight_column)
            best_ams, best_cut = _get_best_ams_cut(*y_pred, *w, bins=bins)
            
            max_ams = np.sum(w[0]) / np.sqrt(np.sum(w[0]))
            
            ams[name].append(best_ams / max_ams)
            cut[name].append(best_cut)
    
    # plot AMS and CUT
    if xticks is None:
        xticks = list(models_and_data.values())[0][-1].unique_signal_mass

    for key, ams_ in ams.items():
        cuts = cut[key]
        
        axes[0].plot(xticks, ams_, marker='o', label=f'{key}: {round(np.mean(ams_).item(), 3)}')
        axes[1].plot(xticks, cuts, marker='o', label=f'{key}: {round(np.mean(cuts).item(), 3)}')
        
    axes[0].set_xlabel('Mass (GeV)')
    axes[0].set_ylabel('Significance / Max Significance')
    axes[0].set_title(f'{title}; #bins = {bins}\nComparison Significance vs mA')
    axes[0].legend(loc='best')
    
    axes[1].set_xlabel('Mass (GeV)')
    axes[1].set_ylabel('Best Cut')
    axes[1].set_title(f'{title}; #bins = {bins}\nComparison Best-Cut vs mA')
    axes[1].legend(loc='best')
    
    fig.tight_layout()
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}.png'), bbox_inches='tight')
    
    plt.show()
    

def cut(models_and_data, title='', bins=20, size=(12, 10), legend='best', name='Model', 
        path='plot', save=None, show=True, ax=None, weight_column='weight', signal_in_interval=False):
    """Plots the value of the best cut as the mass (mA) varies"""
    if ax is None:
        fig = plt.figure(figsize=size)
        ax = fig.gca()
    
    if isinstance(models_and_data, tuple):
        models_and_data = {'': models_and_data}
    
    for key, (model, data) in models_and_data.items():
        cuts = []
        masses = np.sort(data.unique_signal_mass)
        
        # select both signal and background in interval (m-d, m+d)
        for mass, (m_low, m_up) in zip(masses, data.mass_intervals):
            sig = data.signal[data.signal['mA'] == mass]
            
            if signal_in_interval:
                sig = sig[(sig['dimuon_mass'] >= m_low) & (sig['dimuon_mass'] < m_up)]

            bkg = data.background
            bkg = bkg[(bkg['dimuon_mass'] >= m_low) & (bkg['dimuon_mass'] < m_up)]

             # prepare data
            x = pd.concat([sig[data.columns['feature']],
                           bkg[data.columns['feature']]], axis=0).values

            y = np.reshape(pd.concat([sig['type'], bkg['type']], axis=0).values, newshape=[-1])
            x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
            # predict data
            out, y_pred, w = _get_predictions_and_weights(model, x, y, sig, bkg, bins=bins, weight_column=weight_column)
            _, best_cut = _get_best_ams_cut(*y_pred, *w, bins=bins)
            
            cuts.append(best_cut)
        
        # plot
        ax.plot(masses, cuts, marker='o', label=f'{key}: {round(np.mean(cuts).item(), 3)}')
    
    ax.set_xlabel('Mass (GeV)')
    ax.set_ylabel('Best Cut')
    
    ax.legend(loc=str(legend))
    
    # title
    str0 = f'{title} [#bins = {bins}]'
    ax.set_title(f'[{name}] Best Cut vs mA ({str0})')
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}.png'), bbox_inches='tight')
    
    if show:
        plt.show()


def curves(model, dataset: Dataset, mass: int, title='', interval=50, weight_column='weight',
           bins=20, size=(10, 9), legend='best', seed=utils.SEED, path='plot', save=None, signal_in_interval=False):
    """Plots the PR and ROC curves"""
    if isinstance(interval, (int, float)):
        interval = (mass - interval, mass + interval)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    fig.set_figwidth(size[0] * 2)
    fig.set_figheight(size[1])
    
    sig = dataset.signal[dataset.signal['mA'] == mass]

    if signal_in_interval:
        sig = sig[(sig['dimuon_mass'] >= interval[0]) & (sig['dimuon_mass'] < interval[1])]
    
    bkg = dataset.background
    bkg = bkg[(bkg['dimuon_mass'] >= interval[0]) & (bkg['dimuon_mass'] < interval[1])]
    
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
    w_bkg = bkg[weight_column].values
    w_sig = sig[weight_column].values
    
    h_bkg, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)
    h_bkg = np.sum(h_bkg)
    
    h_sig, _ = np.histogram(y_sig, bins=bins)
    h_sig = np.sum(h_sig)
    
    w_sig = w_sig * (h_bkg / h_sig)
    w = np.concatenate([w_sig, w_bkg], axis=0)
    
    str1 = f'@{interval} dimuon_mass (bkg)'
    
    # PR-curve
    PrecisionRecallDisplay.from_predictions(y_true=y, y_pred=out, sample_weight=w, ax=axes[0],
                                            name=f'pNN @ {int(mass)}mA, {str1}')
    axes[0].set_title(f'[pNN] PR Curve @ {int(mass)}mA ({title})')
    
    # ROC curve
    RocCurveDisplay.from_predictions(y_true=y, y_pred=out, sample_weight=w, ax=axes[1],
                                     name=f'pNN @ {int(mass)}mA, {str1}')
    axes[1].set_title(f'[pNN] ROC Curve @ {int(mass)}mA ({title})')
    
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


def compare_roc(dataset: Dataset, models_and_cuts: dict, mass: float, title='', interval=50.0, bins=20, signal_in_interval=False,
                size=(12, 10), digits=3, path='plot', save=None, name='Model', weight_column='weight', ax=None, **kwargs):
    
    if isinstance(interval, (int, float)):
        interval = (mass - interval, mass + interval)

    sig = dataset.signal[dataset.signal['mA'] == mass]

    if signal_in_interval:
        sig = sig[(sig['dimuon_mass'] >= interval[0]) & (sig['dimuon_mass'] < interval[1])]
    
    bkg = dataset.background
    bkg = bkg[(bkg['dimuon_mass'] >= interval[0]) & (bkg['dimuon_mass'] < interval[1])]

    # prepare data
    x = pd.concat([sig[dataset.columns['feature']],
                   bkg[dataset.columns['feature']]], axis=0).values

    y = np.reshape(pd.concat([sig['type'], bkg['type']], axis=0).values, newshape=[-1])
    x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
    # predict data
    if ax is None:
        plt.figure(figsize=size)
        ax = plt.gca()

        should_show = True
    else:
        should_show = False
    
    ax.set_title(f'[{name}] ROC @ {int(mass)}mA [{title}]')

    for k, (model, cut) in models_and_cuts.items():
        out, _, w = _get_predictions_and_weights(model, x, y, sig, bkg, bins, weight_column)
        w = np.concatenate(w, axis=0)

        fpr, tpr, auc, cut_fpr, cut_tpr = roc_auc(true=y, pred=out, weights=w, cut=cut, **kwargs)
    
        ax.plot(fpr, tpr, label=f'AUC ({k}) {np.round(auc, digits)}')
        ax.scatter(cut_fpr, cut_tpr, label=f'significance @ {round(cut, digits)}')
    
    ax.set_xlabel('Background Efficiency (False Positive Rate)')
    ax.set_ylabel('Signal Efficienty (True Positive Rate)')
    
    ax.legend(loc='lower right')
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}_{int(mass)}mA.png'), bbox_inches='tight')
    
    if should_show:    
        plt.show()


def compare_pr(dataset: Dataset, models_and_cuts: dict, mass: float, title='', interval=50.0, bins=20, signal_in_interval=False,
               size=(12, 10), digits=3, path='plot', save=None, name='Model', weight_column='weight', ax=None, **kwargs):    
    if isinstance(interval, (int, float)):
        interval = (mass - interval, mass + interval)

    sig = dataset.signal[dataset.signal['mA'] == mass]

    if signal_in_interval:
        sig = sig[(sig['dimuon_mass'] >= interval[0]) & (sig['dimuon_mass'] < interval[1])]
    
    bkg = dataset.background
    bkg = bkg[(bkg['dimuon_mass'] >= interval[0]) & (bkg['dimuon_mass'] < interval[1])]

    # prepare data
    x = pd.concat([sig[dataset.columns['feature']],
                   bkg[dataset.columns['feature']]], axis=0).values

    y = np.reshape(pd.concat([sig['type'], bkg['type']], axis=0).values, newshape=[-1])
    x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
    # predict data
    if ax is None:
        plt.figure(figsize=size)
        ax = plt.gca()
        
        should_show = True
    else:
        should_show = False
    
    ax.set_title(f'[{name}] PR Curve @ {int(mass)}mA [{title}]')
    
    for k, (model, cut) in models_and_cuts.items():
        out, _, w = _get_predictions_and_weights(model, x, y, sig, bkg, bins, weight_column)
        w = np.concatenate(w, axis=0)
        
        precision, recall, auc, cut_prec, cut_rec = pr_auc(true=y, pred=out, weights=w, cut=cut, **kwargs)
        
        ax.plot(recall, precision, label=f'AUC ({k}) {np.round(auc, digits)}')
        ax.scatter(cut_rec, cut_prec, label=f'significance @ {round(cut, digits)}')
    
    ax.set_xlabel('Recall (signal efficiency)')
    ax.set_ylabel('Precision (purity)')
    
    ax.legend(loc='lower left')
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}_{int(mass)}mA.png'), bbox_inches='tight')
    
    if should_show:
        plt.show()


def var_priori(dataset: Dataset, variables: list, mass: float, interval=50.0, size=(12, 10), 
               legend='best', bins=25, weight_column='weight', path='plot', signal_in_interval=False,
               seed=utils.SEED, save=None, min_limit=None, max_limit=None, palette=PALETTE):
    if isinstance(interval, (int, float)):
        interval = (mass - interval, mass + interval)

    sig = dataset.signal[dataset.signal['mA'] == mass]
    bkg = dataset.background
    
    if signal_in_interval:
        sig = sig[(sig['dimuon_mass'] >= interval[0]) & (sig['dimuon_mass'] < interval[1])]

    bkg = bkg[(bkg['dimuon_mass'] >= interval[0]) & (bkg['dimuon_mass'] < interval[1])]
    
    w_b = bkg[weight_column].values
    w_s = sig[weight_column].values
    w_s = w_s * (np.sum(w_b) / np.sum(w_s))

    names = np.squeeze(bkg['name'])

    for col in variables:
        fig = plt.figure(figsize=size)
        ax = fig.gca()

        s = sig[col]
        b = bkg[col]
        
        df = pd.DataFrame({col: b, 'Bkg': names, 'weight': w_b})

        if ('met' in col) or ('pt' in col):
            n_bins = 50
            max_limit = 500.0

        elif col == 'ljet_n':
            n_bins = 25
            max_limit = 11.0

        elif col == 'dimuon_mass':
            n_bins = 25
            max_limit = interval[1]
            min_limit = interval[0]
        else:
            n_bins = bins
            max_limit = None

        range_min = min(b.min(), s.min())
        range_max = max(b.max(), s.max())

        if isinstance(min_limit, (int, float)):
            range_min = max(range_min, min_limit)

        if isinstance(max_limit, (int, float)):
            range_max = min(range_max, max_limit)

        # plot histograms
        sns.histplot(data=df, x=col, hue='Bkg', multiple='stack', edgecolor='.3', linewidth=0.5, bins=n_bins,
                     weights='weight', ax=ax, binrange=(range_min, range_max), palette=palette)

        ax.hist(s, bins=n_bins, alpha=0.7, label='signal', hatch='//', color=palette['signal'], 
                edgecolor=palette['signal'], linewidth=2,  histtype='step', range=(range_min, range_max), 
                weights=w_s)

        leg = ax.get_legend()
        ax.legend(loc='upper left')
        ax.add_artist(leg)

        ax.set_title(f'mA @ {int(mass)} - dimuon_mass in {interval}')
        ax.set_xlabel(col)
        ax.set_ylabel('Weighted Count')
        
        fig.tight_layout()
    
    if isinstance(save, str):
        path = utils.makedir(path)
        plt.savefig(os.path.join(path, f'{save}_{col}_{int(mass)}_mA.png'), bbox_inches='tight')
    
    plt.show()


def var_posteriori(dataset: Dataset, models: list, variables: list, mass: float, cuts: list, signal_in_interval=False,
                   interval=50.0, size=(12, 10), legend='best', bins=25, share_y=True, weight_column='weight',
                   path='plot', seed=utils.SEED, save=None, min_limit=None, max_limit=None, palette=PALETTE):
    if not isinstance(models, list):
        models = [models]

    if not isinstance(cuts, list):
        cuts = [cuts]

    assert len(models) <= 2, 'At most two models can be compared!'

    def predict(model, x, cut):
        out = model.predict(x=x, batch_size=1024, verbose=0)
        out = np.asarray(out)
    
        sig_mask = np.squeeze(out >= cut) & (y == 1.0)
        bkg_mask = np.squeeze(out < cut) & (y == 0.0)

        y_sig = np.squeeze(out[sig_mask])
        y_bkg = np.squeeze(out[bkg_mask])

        w_bkg = ds[bkg_mask][weight_column].values
        w_sig = ds[sig_mask][weight_column].values

        # compute signal weights
        h_bkg, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)
        h_bkg = np.sum(h_bkg)

        h_sig, _ = np.histogram(y_sig, bins=bins)
        h_sig = np.sum(h_sig)

        w_sig = w_sig * (h_bkg / h_sig)
        names = dataset.names_df.loc[ds[bkg_mask].index]
        
        return out, (sig_mask, bkg_mask), (y_sig, y_bkg), (w_sig, w_bkg), names 
    
    if isinstance(interval, (int, float)):
        interval = (mass - interval, mass + interval)

    sig = dataset.signal[dataset.signal['mA'] == mass]
    bkg = dataset.background
    
    if signal_in_interval:
        sig = sig[(sig['dimuon_mass'] >= interval[0]) & (sig['dimuon_mass'] < interval[1])]

    bkg = bkg[(bkg['dimuon_mass'] >= interval[0]) & (bkg['dimuon_mass'] < interval[1])]
    
    ds = pd.concat([sig, bkg], axis=0)
    num_sig = sig.shape[0]
    
    # prepare data
    x = ds[dataset.columns['feature']].values

    y = np.reshape(pd.concat([sig['type'], bkg['type']], axis=0).values, newshape=[-1])
    x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
    # predict data
    preds = [predict(model, x, cut) for model, cut in zip(models, cuts)]

    out = [p[0] for p in preds]
    mask = [p[1] for p in preds]
    labels = [p[2] for p in preds]
    w = [p[3] for p in preds]
    names = [p[4] for p in preds]

    for col in variables:
        fig, axes = plt.subplots(nrows=1, ncols=len(models), sharey=bool(share_y))
        
        fig.set_figwidth(size[0] * len(models))
        fig.set_figheight(size[1])
        
        if len(models) == 1:
            axes = [axes]

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

            elif col == 'dimuon_mass':
                n_bins = 25
                max_limit = interval[1]
                min_limit = interval[0]
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
                         weights='weight', ax=ax, binrange=(range_min, range_max), palette=palette)

            ax.hist(s, bins=n_bins, alpha=0.7, label='signal', hatch='//', color=palette['signal'], 
                    edgecolor=palette['signal'], linewidth=2,  histtype='step', range=(range_min, range_max), 
                    weights=w[i][0])

            leg = ax.get_legend()
            ax.legend(loc='upper left')
            ax.add_artist(leg)

            ax.set_title(f'mA @ {int(mass)} - dimuon_mass in {interval}')
            ax.set_xlabel(col)
            ax.set_ylabel('Weighted Count')
            
            fig.tight_layout()
        
        if isinstance(save, str):
            path = utils.makedir(path)
            plt.savefig(os.path.join(path, f'{save}_{col}_{int(mass)}_mA.png'), bbox_inches='tight')
        
        plt.show()


def variables(dataset: Dataset, model, variables: list, mass: float, cut: list, interval=50.0, size=(12, 10), 
              legend='best', bins=25, share_y=True, weight_column='weight', path='plot', seed=utils.SEED, 
              save=None, min_limit=None, max_limit=None, palette=PALETTE, signal_in_interval=False):
    """Variables at priori (before applying NN) vs poteriori (after applying NN)"""
    def predict(model, x, cut):
        out = model.predict(x=x, batch_size=1024, verbose=0)
        out = np.asarray(out)
    
        sig_mask = np.squeeze(out >= cut) & (y == 1.0)
        bkg_mask = np.squeeze(out < cut) & (y == 0.0)

        y_sig = np.squeeze(out[sig_mask])
        y_bkg = np.squeeze(out[bkg_mask])

        w_bkg = ds[bkg_mask][weight_column].values
        w_sig = ds[sig_mask][weight_column].values

        # compute signal weights
        h_bkg, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)
        h_bkg = np.sum(h_bkg)

        h_sig, _ = np.histogram(y_sig, bins=bins)
        h_sig = np.sum(h_sig)

        w_sig = w_sig * (h_bkg / h_sig)
        names = dataset.names_df.loc[ds[bkg_mask].index]
        
        return out, (sig_mask, bkg_mask), (y_sig, y_bkg), (w_sig, w_bkg), names 
    
    if isinstance(interval, (int, float)):
        interval = (mass - interval, mass + interval)

    sig = dataset.signal[dataset.signal['mA'] == mass]
    bkg = dataset.background
    
    if signal_in_interval:
        sig = sig[(sig['dimuon_mass'] >= interval[0]) & (sig['dimuon_mass'] < interval[1])]

    bkg = bkg[(bkg['dimuon_mass'] >= interval[0]) & (bkg['dimuon_mass'] < interval[1])]
    
    ds = pd.concat([sig, bkg], axis=0)
    norm_w = np.sum(bkg[weight_column]) / np.sum(sig[weight_column])
    
    # prepare data
    x = ds[dataset.columns['feature']].values

    y = np.reshape(pd.concat([sig['type'], bkg['type']], axis=0).values, newshape=[-1])
    x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)
    
    # predict data
    _, (s_mask, b_mask), _, (w_s, w_b), names = predict(model, x, cut)

    for col in variables:
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=bool(share_y))
        
        fig.set_figwidth(size[0] * 2)
        fig.set_figheight(size[1])
        
        for i, ax in enumerate(axes):
            if i == 0:
                # var-priori: don't apply classifier
                s = sig[col]
                b = bkg[col]
                n = np.squeeze(bkg['name'])
                
                wb = bkg[weight_column]
                ws = sig[weight_column] * norm_w
            else:
                # var-posteriori: apply classifier
                s = ds[s_mask][col]
                b = ds[b_mask][col]
                n = np.squeeze(names)
                
                wb = w_b
                ws = w_s

            df  = pd.DataFrame({col: b, 'Bkg': n, 'weight': wb})

            if ('met' in col) or ('pt' in col):
                n_bins = 50
                max_limit = 500.0

            elif col == 'ljet_n':
                n_bins = 25
                max_limit = 11.0

            elif col == 'dimuon_mass':
                n_bins = 25
                max_limit = interval[1]
                min_limit = interval[0]
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
                         weights='weight', ax=ax, binrange=(range_min, range_max), palette=palette)

            ax.hist(s, bins=n_bins, alpha=0.7, label='signal', hatch='//', color=palette['signal'], 
                    edgecolor=palette['signal'], linewidth=2,  histtype='step', range=(range_min, range_max), 
                    weights=ws)

            leg = ax.get_legend()
            ax.legend(loc='upper left')
            ax.add_artist(leg)

            title = f'{"priori" if i == 0 else "posteriori@" + str(round(cut, 3))}' 
            ax.set_title(f'[{title}] mA @ {int(mass)} - dimuon_mass in {interval}')
            
            ax.set_xlabel(col)
            ax.set_ylabel('Weighted Count')
            
            fig.tight_layout()
        
        if isinstance(save, str):
            path = utils.makedir(path)
            plt.savefig(os.path.join(path, f'{save}_{col}_{int(mass)}_mA.png'), bbox_inches='tight')
        
        plt.show()


def curve_vs_mass(models_and_data: dict, bins=20, size=(12, 10), path='plot', save=None, title='pNN', 
                  auc=False, curve='roc', legend='best', weight_column='weight', signal_in_interval=False, **kwargs):
    is_roc = curve.lower() == 'roc'
    
    if isinstance(models_and_data, tuple):
        models_and_data = {'': models_and_data}
    
    if auc:
        curve = {name: [] for name in models_and_data.keys()} 
    
    elif is_roc:
        curve = {name: dict(FPR=[], TPR=[]) for name in models_and_data.keys()}
    else:
        curve = {name: dict(precision=[], recall=[]) for name in models_and_data.keys()}
    
    for name, (model, dataset) in models_and_data.items():
        for mass, (m_low, m_up) in zip(dataset.unique_signal_mass, dataset.mass_intervals):
            sig = dataset.signal
    
            # select data
            s = sig[sig['mA'] == mass]

            if signal_in_interval:
                s = s[(s['dimuon_mass'] >= m_low) & (s['dimuon_mass'] < m_up)]

            b = dataset.background
            b = b[(b['dimuon_mass'] >= m_low) & (b['dimuon_mass'] < m_up)]

            # prepare data
            x = pd.concat([s[dataset.columns['feature']],
                           b[dataset.columns['feature']]], axis=0).values

            y = np.reshape(pd.concat([s['type'], b['type']], axis=0).values, newshape=[-1])
            x = dict(x=x, m=np.ones_like(y[:, np.newaxis]) * mass)

            # compute curve @ cut
            out, y_pred, w = _get_predictions_and_weights(model, x, y, sig=s, bkg=b, bins=bins, weight_column=weight_column)
            _, cut = _get_best_ams_cut(*y_pred, *w, bins=bins)
            w = np.concatenate(w, axis=0)

            if is_roc:
                _, _, auc_, cut_fpr, cut_tpr = roc_auc(true=y, pred=out, weights=w, cut=cut, **kwargs)

                if auc:
                    curve[name].append(auc_)
                else:
                    curve[name]['FPR'].append(cut_fpr)
                    curve[name]['TPR'].append(cut_tpr)
            else:
                _, _, auc_, cut_prec, cut_rec = pr_auc(true=y, pred=out, weights=w, cut=cut, **kwargs)

                if auc:
                    curve[name].append(auc_)
                else:
                    curve[name]['precision'].append(cut_prec)
                    curve[name]['recall'].append(cut_rec)
    # plot
    plt.figure(figsize=size)
    plt.title(f'[{title}] {"ROC" if is_roc else "PR"} Curve @ Best-Cut vs mA')
    
    ax = plt.gca()
    
    if not auc:
        bx = plt.twinx()
        bx.grid(False)
        
        for name, (_, dataset) in models_and_data.items():
            mass = dataset.unique_signal_mass
            
            for key, value in curve[name].items():
                plt.plot(mass, value, marker='o', label=f'{key}-{name}: {round(np.mean(value).item(), 2)}')
    else:
        for name, (_, dataset) in models_and_data.items():
            mass = dataset.unique_signal_mass
            
            plt.plot(mass, curve[name], marker='o', label=f'AUC-{name}: {round(np.mean(curve[name]).item(), 2)}')
    
    ax.set_xlabel('Mass (GeV)')
    
    if auc:
        ax.set_ylabel('AUC')
    
    elif is_roc:
        bx.set_ylabel('Signal Efficiency (TPR)')
        ax.set_ylabel('Background Efficiency (FPR)')
    else:
        bx.set_ylabel('Precision (purity)')
        ax.set_ylabel('Recall (signal efficiency)')
    
    plt.legend(loc=str(legend).lower())
    plt.show()

    
def _get_predictions_and_weights(model, x, y, sig: pd.DataFrame, bkg: pd.DataFrame, bins, weight_column='weight'):
    out = model.predict(x=x, batch_size=1024, verbose=0)
    out = np.asarray(out)

    y_sig = np.squeeze(out[y == 1.0])
    y_bkg = np.squeeze(out[y == 0.0])

    # compute weights
    w_bkg = bkg[weight_column].values
    w_sig = sig[weight_column].values

    h_bkg, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)
    h_bkg = np.sum(h_bkg)

    h_sig, _ = np.histogram(y_sig, bins=bins)
    h_sig = np.sum(h_sig)

    w_sig = w_sig * (h_bkg / h_sig)
    
    return out, (y_sig, y_bkg), (w_sig, w_bkg)

    
def _get_best_ams_cut(y_sig, y_bkg, w_sig, w_bkg, bins):
    cuts = np.linspace(0.0, 1.0, num=bins)
    ams = []

    s, _ = np.histogram(y_sig, bins=bins, weights=w_sig)
    b, _ = np.histogram(y_bkg, bins=bins, weights=w_bkg)

    for i in range(s.shape[0]):
        s_i = np.sum(s[i:])
        b_i = np.sum(b[i:])

        ams.append(s_i / np.sqrt(s_i + b_i))

    return np.max(ams), cuts[np.argmax(ams)]
