import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap

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



# copyright unknown :) believe it is public domain
def plot_decision_regions(X, y, 
                          classifier=None, 
                          label_y='class=',
                          labels_text=None,
                          cmap=None,
                          ax=None):
    
    labels = np.unique(y)
    markers = [ 's', 'o', '^', 'v', 'x' ]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    cmap = ListedColormap( colors[:len(labels)] ) if not cmap else cmap
    ax = plt.gca() if not ax else ax 
    
    x1_min, x2_min = X.min(0) - 0.25
    x1_max, x2_max = X.max(0) + 0.25
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

    plt.grid()
    
    # decision areas
    if classifier:
        x1r = np.linspace(x1_min, x1_max, 400)
        x2r = np.linspace(x2_min, x2_max, 400)
        xx1, xx2 = np.meshgrid(x1r, x2r)
        ar = np.array([xx1.ravel(), xx2.ravel()]).T
        
        if classifier.__class__.__name__=='SVC':
            Z = classifier.decision_function(ar).reshape(xx1.shape)
        else:
            Z = classifier.predict(ar).reshape(xx1.shape)
            
        cp = ax.contour( xx1, xx2, Z, 
                    levels=[ -1, 0, 1, ],
                    linestyles=['--', '-', '--',],
                    cmap=cmap)
        ax.clabel(cp, inline=1, fontsize=10)
        ax.contourf( xx1, xx2, Z, 
                    alpha=0.1, 
                    levels=[-100500, -1, 0, 1, 100500],
                    cmap=cmap)

    # samples
    for k, cl in enumerate(labels):
        idx = ( y == cl )
        label = label_y+str(labels_text[k] if labels_text else cl) if label_y else None
        ax.scatter(x=X[idx,0], y=X[idx,1],
                    alpha=0.8, c=colors[k], marker=markers[k],
                    label=label, edgecolor='black')
        
        
def plot_with_err(x, data, **kwargs):
    mu, std = data.mean(1), data.std(1)
    lines = plt.plot(x, mu, '-', **kwargs)
    plt.fill_between(x, mu - std, mu + std, edgecolor='none',
                     facecolor=lines[0].get_color(), alpha=0.2)