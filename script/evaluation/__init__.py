"""Evaluation Procedures"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from script.evaluation import hepmass

from script import utils
from script.utils import assert_2d_array

from typing import Union, List


def mean_mass(intervals):
    assert_2d_array(intervals)
    return np.mean(intervals, axis=1)


def check_underflow(metric, control_metric, threshold=0.5) -> float:
    if metric == 0.0 and control_metric > threshold:
        print(f'[Probable underflow] metric: {metric}, control-metric: {control_metric}')
        return 1.0  # underflow occurred (metric would be wrongly 0.0)
    
    return metric  # no underflow


def split_data(data: tuple, num_folds=10, seed=utils.SEED):
    """Splits a given Dataframe into k disjoint folds"""
    fold_size = data[1].shape[0] // num_folds
    folds = []
    
    num_features = data[0]['x'].shape[-1]
    column_names = [str(i) for i in range(num_features)]
    
    labels = data[1]
    is_multi_label = labels.shape[-1] > 1
    
    if is_multi_label:
        label_df = pd.DataFrame({'label': list(data[1])})
    else:
        label_df = pd.DataFrame(data[1], columns=['label'])
    
    # first, construct a dataframe from data, where `data = dict(x, m), y`
    df = pd.concat([
        pd.DataFrame(data[0]['x'], columns=column_names),
        pd.DataFrame(data[0]['m'], columns=['mass']),
        label_df
    ], axis=1)
    
    for _ in range(num_folds - 1):
        fold = df.sample(fold_size, random_state=seed)
        folds.append(fold)
        
        df.drop(fold.index, inplace=True)
    
    folds.append(df)
    
    # make each `fold` be structured like `data`, i.e. fold = tuple(dict(x, m), y)
    for i, fold in enumerate(folds):       
        fold_x = dict(x=fold[column_names].values, 
                      m=fold['mass'].values)
        
        if is_multi_label:
            fold_y = np.stack(fold['label'].values)
        else:
            fold_y = fold['label'].values
    
        folds[i] = (fold_x, fold_y)
    
    return folds


def metric_with_error(model, dataset, metric: str, index: int, mass_intervals: Union[np.ndarray, List[tuple]] = None,
                      figsize=(26, 20), num_folds=10, verbose=0, silent=False, style='bar', return_average=True, **kwargs):
    """Computes given metric on disjoint folds of the given data, quantifying how much uncertain the predictions are"""
    assert_2d_array(mass_intervals)
    plt.figure(figsize=figsize)

    style = style.lower()
    assert style in ['bar', 'dot', 'fill']

    if mass_intervals is None:
        mass_intervals = dataset.current_mass_intervals

    metric_name = metric
    mass = mean_mass(mass_intervals)
    metric = {fold: [] for fold in range(num_folds)}

    for interval in mass_intervals:
        x, y = dataset.get_by_mass(interval, **kwargs)
        
        folds = split_data(data=(x, y), num_folds=num_folds)
        
        for i, fold in enumerate(folds):
            score = model.evaluate(x=fold[0], y=fold[1], batch_size=128, verbose=verbose)

            metric_score = round(score[index], 4)
            metric[i].append(check_underflow(metric_score, score[1]))  # use accuracy as reference
        
        if not silent:
            print(f'Mass:{np.round(interval, 2)} -> {round(np.mean(interval))}')
    
    # compute average metric (over folds)
    avg_metric = []
    
    for i, _ in enumerate(mass):
        score = 0.0
        
        for fold in range(num_folds):
            score += metric[fold][i]
        
        avg_metric.append(round(score / num_folds, 4))

    plt.title(f'{metric_name} vs Mass')
    plt.ylabel(metric_name)
    plt.xlabel('Mass')
    
    if style == 'dot': 
        plt.plot(mass, avg_metric, label='avg')
        plt.scatter(mass, avg_metric, s=50, color='b')
        
        for i in range(num_folds):
            plt.scatter(mass, metric[i], s=30, color='r')
    else:
        values = np.array(list(metric.values()))
        avg_metric = np.array(avg_metric)

        if style == 'bar':
            min_err = avg_metric - np.min(values, axis=0)
            max_err = np.max(values, axis=0) - avg_metric

            plt.errorbar(mass, avg_metric, yerr=np.stack([min_err, max_err]), fmt='ob', 
                         capsize=5.0, elinewidth=1, capthick=1)
            plt.plot(mass, avg_metric, label='avg')
        else:
            min_err = np.min(values, axis=0)
            max_err = np.max(values, axis=0)

            plt.fill_between(mass, min_err, max_err, color='gray', alpha=0.2)

            plt.plot(mass, avg_metric, label='avg')
            plt.scatter(mass, avg_metric, s=50, color='b')
    
    plt.show()

    if return_average:
        return np.mean(list(metric.values()), axis=0)

    return metric


def auc_with_error(model, dataset, index=2, mass_intervals: Union[np.ndarray, List[tuple]] = None,
                   figsize=(26, 20), num_folds=10, verbose=0, silent=False, style='bar', return_average=True, **kwargs):
    """Computes AUC on disjoint folds of the given data, quantifying how much uncertain the predictions are"""

    return metric_with_error(model, dataset, metric='AUC', index=index, mass_intervals=mass_intervals, figsize=figsize,
                             num_folds=num_folds, verbose=verbose, silent=silent, style=style, return_average=return_average, **kwargs)


def auc_vs_no_mass(model, dataset, auc_index=2, mass_intervals: Union[np.ndarray, List[tuple]] = None, 
                   fake_mass=[], figsize=(26, 20), verbose=0, silent=False, sample_frac=None, avg_auc=None, **kwargs):
    """Computes AUC by faking the true mass"""
    assert_2d_array(mass_intervals)
    plt.figure(figsize=figsize)
    
    if mass_intervals is None:
        mass_intervals = dataset.current_mass_intervals

    # scale mass
    if dataset.m_scaler is not None:
        scaled_fake_mass = dataset.m_scaler.transform(np.reshape(fake_mass, newshape=(-1, 1)))
        scaled_fake_mass = np.squeeze(scaled_fake_mass)
    else:
        scaled_fake_mass = fake_mass
    
    mass = mean_mass(mass_intervals)
    auc = {m: [] for m in scaled_fake_mass}

    for interval in mass_intervals:
        if not silent:
            print(f'Mass interval {interval} -> {round(np.mean(interval), 2)}')

        x, y = dataset.get_by_mass(interval, sample=sample_frac, **kwargs)
        
        for fake_m in scaled_fake_mass:
            x['m'] = np.ones_like(x['m']) * fake_m
        
            score = model.evaluate(x=x, y=y, batch_size=128, verbose=verbose)
            
            auc_score = round(score[auc_index], 4)
            auc[fake_m].append(check_underflow(auc_score, score[1]))
    
    plt.title(f'AUC vs Mass')
    
    if isinstance(avg_auc, (list, np.ndarray)):
        plt.plot(mass, avg_auc, label='auc')

    for i, fake_m in enumerate(fake_mass):
        k = scaled_fake_mass[i]
        
        plt.plot(mass, auc[k], label=f'm-{round(fake_m, 2)}')
        plt.scatter(mass, auc[k], s=50)
        
    plt.legend(loc='best')
    plt.ylabel('AUC')
    plt.xlabel('Mass')
    
    plt.show()
    return auc


def auc_vs_mass_no_features(model, dataset, auc_index=2, mass_intervals: Union[np.ndarray, List[tuple]] = None, 
                            figsize=(26, 20), features={}, verbose=1, silent=False, sample_frac=None, 
                            transformer=None, transform_before=True):
    """Computes AUC by dropping one or more features"""
    assert_2d_array(mass_intervals)
    plt.figure(figsize=figsize)

    if mass_intervals is None:
        mass_intervals = dataset.current_mass_intervals

    mass = mean_mass(mass_intervals)
    auc = {k: [] for k in features.keys()}
    
    for label, indexes in features.items():
        if not silent:
            print(f'Features: {label}, {indexes}')
        
        for interval in mass_intervals:
            if transform_before:
                x, y = dataset.get_by_mass(interval, sample=sample_frac, transformer=transformer)
            else:
                x, y = dataset.get_by_mass(interval, sample=sample_frac)
        
            # mask features
            for i in indexes:
                zero_feature = np.zeros_like(x['x'][:, i])
                x['x'][:, i] = zero_feature
            
            if (not transform_before) and (transformer is not None):
                x['x'] = transformer.transform(x['x'])
            
            if not silent:
                print(f'Mass: {np.round(interval, 2)} -> {round(np.mean(interval), 2)}')
    
            score = model.evaluate(x=x, y=y, batch_size=128, verbose=verbose)
            
            auc_score = round(score[auc_index], 4)
            auc[label].append(check_underflow(auc_score, score[1]))
            
    plt.title(f'AUC vs Mass')
    
    for label in features.keys():
        plt.plot(mass, auc[label])
        plt.scatter(mass, auc[label], s=50, label=label)
    
    plt.ylabel('AUC')
    plt.xlabel('Mass')
    
    plt.legend()
    plt.show()
    return auc


def auc_mass_importance(model, dataset, auc_index: int, mass: list, mass_intervals: Union[np.ndarray, List[tuple]] = None,
                        figsize=(26, 20), verbose=0, silent=False, reference=None, **kwargs):
    """Computes AUC on disjoint folds of the given data, quantifying how much the mass helps the classification"""
    assert_2d_array(mass_intervals)
    assert isinstance(mass, (list, tuple, np.ndarray))

    plt.figure(figsize=figsize)

    if mass_intervals is None:
        mass_intervals = dataset.current_mass_intervals

    if dataset.m_scaler is not None:
        scaled_mass = dataset.m_scaler.transform(np.reshape(mass, newshape=(-1, 1)))
        scaled_mass = np.reshape(scaled_mass, newshape=[-1])
    else:
        scaled_mass = mass

    avg_mass = mean_mass(mass_intervals)
    auc = {m: [] for m in scaled_mass}
    
    for interval in mass_intervals: 
        x, y = dataset.get_by_mass(interval, **kwargs)
        
        for j, m in enumerate(scaled_mass):    
            x['m'] = np.ones_like(x['m']) * m

            score = model.evaluate(x, y, batch_size=128, verbose=verbose)

            auc_score = round(score[auc_index], 4)
            auc[m].append(check_underflow(auc_score, score[1]))
            
        if not silent:
            print(f'Mass:{np.round(interval, 2)} -> {round(np.mean(interval))}')
    
    plt.title(f'AUC vs Mass')
    plt.ylabel('AUC')
    plt.xlabel('Mass')
    
    if isinstance(reference, (list, tuple, np.ndarray)):
        plt.plot(avg_mass, reference, label='auc')

    for i, m in enumerate(scaled_mass):
        plt.plot(avg_mass, auc[m], label=f'm-{round(mass[i], 1)}')
        plt.scatter(avg_mass, auc[m], s=50)

    plt.legend(loc='best')
    plt.show()

    return auc
