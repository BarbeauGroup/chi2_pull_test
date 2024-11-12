import numpy as np

def marginalize(cost_function, x, i, j, i_range, j_range):
    chi2_arr = np.zeros((len(i_range), len(j_range)))
    for i_idx, i_val in enumerate(i_range):
        for j_idx, j_val in enumerate(j_range):
            x[i] = i_val
            x[j] = j_val
            chi2_arr[i_idx, j_idx] = cost_function(x)
    return chi2_arr
