import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_outliers(data, ax, lower_bound=None, upper_bound=None, method=None, z_thres=3, **kwargs):

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    if lower_bound is None or upper_bound is None:
        if method == 'tukey':
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        elif method=='z-score':
            mean = data.mean()
            std = data.std()
            lower_bound = mean - z_thres * std
            upper_bound = mean + z_thres * std
    
    sns.histplot(data, ax=ax, bins=kwargs.get('bins', 'auto'), log_scale=kwargs.get('logscale', False))
    if method=='tukey':
        ax.axvline(data.median(), color='r', linestyle='--', label=f'Медиана: {data.median():.2f}')
    elif method=='z-score':
        ax.axvline(data.mean(), color='r', linestyle='--', label=f'Среднее: {data.mean():.2f}')
    ax.axvline(Q1, color='r', linestyle=':', label=f'Q1: {Q1:.2f}')
    ax.axvline(Q3, color='r', linestyle=':', label=f'Q3: {Q3:.2f}')

    ax.axvspan(data.min(), lower_bound, alpha=0.1, color='red', label=f'Выбросы < {lower_bound:.2f}')
    ax.axvspan(upper_bound, data.max(), alpha=0.1, color='red', label=f'Выбросы > {upper_bound:.2f}')
    ax.set_xlim((data.min(), data.max()))

    # к фильтрации:
    mask = (data < lower_bound) | (data > upper_bound)
    ax.set_title(f'Выбросы по {method.title()}: {mask.sum()} из {data.shape[0]} ({mask.sum()/data.shape[0]:.2%})')

    ax.legend()