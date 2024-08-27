'''Predict loss from given set of variables (e.g., depth, width, tokens, or parameters & tokens)'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.special import huber
from itertools import product
from configs import *
from utils import *

def log_pred(exps, coefs, e: float, data) -> np.ndarray | float:
    """Predict the log-loss from the power law given the parameter values for exponents and coefficients, and irreducible error e
        
        Adapting log_sum_exp from the Chinchilla replication to handle multiple rows and variations in setup.
        Note e is exponentiated, which presumably is done to keep irreducible error non-negative.
        data: 1 or 2D array"""
    # print('ln(data)', np.log(data))
    # print('exps * ln(data)', exps * np.log(data))
    # print('coefs - exps * ln(data)', coefs - exps * np.log(data))
    # print('sum(exp(coefs - exps * ln(data)))', np.sum(np.exp(coefs - exps * np.log(data)), axis=-1))
    return np.log(np.sum(np.exp(coefs - exps * np.log(data)), axis=-1) + np.exp(e))

def huber_loss_objective(params, data, losses):
    """Also adapted from Chinchilla replication"""
    exps, coefs, e = params
    pred = log_pred(exps, coefs, e, data)
    return np.sum(huber(pred - losses))

def perform_main_analysis(results_df, configs,
                          seed=42, *, seed_noise_args=None, 
                          keep_bs_lr_keys=False
                          ):
    np.random.seed(seed)

    if seed_noise_args is None:
        seed_noise_args = SEED_ARGS
    df = results_df.copy()
    out = []
    for config in configs:
        dataset, hparams, warmup, decay, param_count, val = config
        show_df = df.query(f"dataset=='{dataset}' and hparams=='{hparams}' and warmup=='{warmup}' and decay=='{decay}'")
        show_df['t'] = df['val/loss'].map(lambda x: x.index[-1]) * df.bs * df.seq_len
        show_df['final_loss'] = df['val/loss'].map(lambda x: x.iloc[-1])
        # show_df['t'] = show_df. / show_df.flops_per_token

        if len(show_df) == 0:
            continue

        
        # data, optimal_pairs, max_loss, min_loss = interp_flop(
        #     show_df, seed_noise = seed_noise_args[config], 
        #     flop_vals=flop_vals, **ISOFLOP_ARGS[config[-2:]],
        #     keep_bs_lr_keys=keep_bs_lr_keys,
        # )

        # fit_results = fit_compute_optimal_power_laws(optimal_pairs, data, fit_loss=fit_loss)
        init_params = (np.random.uniform(0, 2.5, 3), np.random.uniform(0, 30, 3), np.random.uniform(-1, 1.5, 1))
        
        fit_results = minimize(huber_loss_objective, init_params, args=(show_df[['width', 'depth', 't']], show_df.final_loss))

        out.append(dict(dataset=dataset, hparams=hparams, warmup=warmup, decay=decay, param_count=param_count, val=val, 
                        fit_results=fit_results,
                        )) # optimal_pairs=optimal_pairs, data=data, max_loss=max_loss, min_loss=min_loss,))
    return pd.DataFrame(out)