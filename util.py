import numpy as np

class Accumulator():
    def __init__(self):
        self.data = {}
        self.step = 0
        
    def append(self, step=None, **kwargs):
        if step is None:
            pass
        for key, val in kwargs.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(val)
            
    def keys(self):
        return self.data.keys()
        
    def get_dict_list(self):
        return self.data
    
    def get_dict_numpy(self):
        return {key: np.array(val) for key, val in self.data.items()}
    
    def get_dict_avg(self):
        return {key: np.array(val).mean() for key, val in self.data.items()}
    
    def __getitem__(self, key):
        return np.array(self.data[key])
    
    
import matplotlib.pyplot as plt
    
def plot_accumulators(accus, keys=None, nrows=None, ncols=None, plot_over_keys=True):
    if isinstance(accus, list):
        accus = {i: accu for i, accu in enumerate(accus)}
        
    if keys is None:
        keys = set()
        for accu in accus.values():
            keys.update(set(accu.keys()))
    
    if plot_over_keys:
        if nrows is None and ncols is None:
            nrows, ncols = len(keys), 1
        for idx_key, key in enumerate(keys):
            plt.subplot(nrows, ncols, idx_key+1)
            for accu_name, accu in accus.items():
                if key in accu.keys():
                    plt.plot(accu[key], label=f'{accu_name}: {key}')
            plt.ylabel(f'{key}')
            plt.xlabel(f'time steps')
            plt.legend()
    else: # plot over accus
        if nrows is None and ncols is None:
            nrows, ncols = len(accus), 1
        for idx_accu, (accu_name, accu) in enumerate(accus.items()):
            plt.subplot(nrows, ncols, idx_accu+1)
            for idx_key, key in enumerate(keys):
                if key in accu.keys():
                    plt.plot(accu[key], label=f'{accu_name}: {key}')
            plt.ylabel(f'{accu_name}')
            plt.xlabel(f'time steps')
            plt.legend()
            
            
    plt.tight_layout()