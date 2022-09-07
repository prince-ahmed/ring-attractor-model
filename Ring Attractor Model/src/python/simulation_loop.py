from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from tqdm import tqdm
from ring_attractor import RingAttractor
from time import time



params = {
    "neurons_n": 256,
    "simulation_time": 10000,
    "iterations": 50,
    "noise_levels": 300,
    "noise_low": 0.0,
    "noise_high": 3e-3,
    "weights": [0.050, 0.100, 0.050, 0.250],  # ext, inh, fp ext, inh
    "fixed_points": [0, 1, 2, 4, 8, 16, 32]
}


def simulation(parameters, _noise, noise_idx, _fp_n, it_n):

    ring = RingAttractor(n=parameters["neurons_n"],
                        noise=_noise,
                        weights=parameters["weights"],
                        fixed_points_number=_fp_n,
                        time=parameters["simulation_time"],
                        # random_seed=seeds[it_n]
                        )

    e = ring.simulate()

    return e, noise_idx, _fp_n, it_n

if __name__ == '__main__':
    noises = np.linspace( params["noise_low"], params["noise_high"], params["noise_levels"])
    noises_idx = ["{:.2e}".format(i) for i in noises]

    records = [pd.DataFrame(index=params["fixed_points"], columns=noises_idx) for _ in range(params["iterations"])]
    seeds = np.random.choice(10000, params["iterations"])
    futures = []


    start = time()
    with ProcessPoolExecutor() as executor:

        for it in range(params["iterations"]):
            for noise_idx, noise in zip(noises_idx, noises):
                for fp_n in params["fixed_points"]:

                    futures.append(executor.submit(simulation, params, noise, noise_idx, fp_n, it))

        for f in tqdm(futures):
            error, noise_idx, fp_idx, it_n = f.result()
            records[it_n].loc[fp_idx, noise_idx] = error


    end = time()
    print("time elapsed: ", end-start)
    for i, df in enumerate(records):
        df.to_csv("csv/seeds/seed_{}.csv".format(seeds[i]))

    full_df = pd.concat(records).astype(float)
    df_average = full_df.groupby(full_df.index).mean()

    df_average.to_csv("csv/means.csv")
