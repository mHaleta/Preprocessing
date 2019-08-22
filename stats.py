import numpy as np

def get_boot_stats(x,statfunc,iterations=1000,iterative=False):
    min_i = 0
    max_i = x.shape[0]
    if not iterative:
        x_index = np.random.randint(min_i,max_i,(max_i,iterations))
        stat_res = statfunc(x[x_index],axis=0)
        return stat_res
    stat_res = []
    for i in range(iterations):
        x_index = np.random.randint(min_i,max_i,max_i)
        stat_res.append(statfunc(x[x_index]))
    return np.array(stat_res)