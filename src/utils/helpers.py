import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import pandas as pd

def calc_corr_sim(x, y):
    # Calculate mean for all values in x
    x_m = round(np.nanmean(x),2)
    # Calculate mean for all values in y
    y_m = round(np.nanmean(y),2)
    # Create a mask for shared ratings
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) > 1:  # Need at least two common elements for correlation
        # Numerator
        num = np.sum((x[mask] - x_m) * (y[mask] - y_m))
        # Denominator
        den = np.sqrt(sum((x[mask] - x_m)**2)) * np.sqrt(sum((y[mask] - y_m)**2))
        if den > 0:
            return np.divide(num, den)
        else:
            return np.nan
    else:
        return np.nan
    

def calc_cos_sim(u, v):
    # Create a mask for shared ratings
    mask = ~np.isnan(u) & ~np.isnan(v)
    if np.sum(mask) > 0:  # Need at least two common elements for correlation
        return 1 - cosine(u[mask], v[mask])
    else:
        return np.nan
    

def sim_matrix_nan(data, name):
    m = data.shape[0]
    # Initialize the similarity matrix to np.nan
    result = np.full((m, m), np.nan)
    # Iterate over all pairs of columns
    for i in range(m):
        for j in range(i, m):
            if name == 'cosine':
                result[i, j] = calc_cos_sim(data.iloc[i], data.iloc[j])
            elif name == 'pearson':
                result[i, j] = calc_corr_sim(data.iloc[i], data.iloc[j])
            else:
                break
            result[j, i] = result[i, j]
    return pd.DataFrame(result, columns=data.index, index=data.index)


def create_long_data(data, dtypes):
    long_data = (
        data
        .stack()
        .loc[lambda x: x.notna() & (x > 0)]
        .reset_index()
    )

    long_data.columns = list(dtypes.keys())
    long_data = long_data.astype(dtypes)
    return long_data


def gini_index(x):
    unique_labels, counts = np.unique(x, return_counts=True)
    probabilities = counts / len(x)
    gini = 1 - np.sum(probabilities**2)
    return (counts, gini)


def entropy_loss(x):
    unique_labels, counts = np.unique(x, return_counts=True)
    probabilities = counts / len(x)
    entropy_loss = entropy(probabilities, base=2)
    return (counts, entropy_loss)


def weighted_impurity(cond_t, cond_f, samples, criteria):
    if criteria == 'gini':
        imp_t = gini_index(cond_t)
        imp_f = gini_index(cond_f)
    elif criteria == 'entropy':
        imp_t = entropy_loss(cond_t)
        imp_f = entropy_loss(cond_f)
    w_imp_t = sum(imp_t[0]) / len(samples) * imp_t[1]
    w_imp_f = sum(imp_f[0]) / len(samples) * imp_f[1]
    return w_imp_t + w_imp_f