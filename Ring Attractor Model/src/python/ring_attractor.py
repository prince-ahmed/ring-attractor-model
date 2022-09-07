import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import vonmises, entropy
from lif_model import LIF
from tqdm import tqdm


class RingAttractor:
    "A self-contained class for the ring attractor"

    def __init__(self,
                n=256,
                noise=2.0e-3,
                weights=(0.050, 0.100, 0.050, 0.250),
                fixed_points_number=0,
                time=1000,
                plot=False,
                random_seed=None,
                # my stuff
                n_exc_syn=4,
                n_inh_syn=7,
                global_inh=0,
                opto_starting_point=50,
                opto_weight=5,
                opto_stim_begin=50,
                opto_duration=20,
                stim_width=7,
                is_global=False,
                starting_weight=0,
                weight_decay=0,
                max_weight=0
                ):

        self.n = n
        self.noise = noise
        self.weights = weights
        self.fp_n = fixed_points_number
        self.time = time
        self.plot = plot
        self.random_seed = random_seed

        # my stuff
        self.n_exc_syn = n_exc_syn
        self.n_inh_syn = n_inh_syn
        self.global_inh = global_inh
        self.opto_starting_point = opto_starting_point
        self.opto_weight = opto_weight
        self.opto_stim_begin = opto_stim_begin
        self.opto_duration = opto_duration
        self.stim_width = stim_width
        self.is_global = is_global
        self.starting_weight = starting_weight
        self.weight_decay = weight_decay
        self.max_weight = max_weight

        self.neurons = [LIF(ID=i,
                            angle=360.0/n*i,
                            noise_mean=0,
                            noise_std=self.noise) for i in range(n)]
        self.fp_width = 3
        self.fixed_points = self.get_fixed_points()
        self.mid_point = n // 2

        self.connect_with_fixed_points()
        self.flushed = True
        self.raw_data = None
        self.spikes = None

        if random_seed:
            np.random.seed(self.random_seed)

    def simulate(self):
        if self.flushed == False:
            warnings.warn("Simulation has not been flushed!")

        potentials = [[] for _ in range(self.n)]
        
        total_activity = []
        for t in tqdm(range(self.time)):
            # print(t, total_activity)
            total_active_neurons = 0
            for neuron in self.neurons:
                if neuron.V == 0:
                    total_active_neurons += 1

                self.input_source(n_of_spikes=self.opto_weight, starting_point=self.opto_starting_point, begin=self.opto_stim_begin, duration=self.opto_duration, neuron=neuron, time=t)
                if t == 0:
                    if neuron.id in range(31, 36):
                        neuron.V = -0.0001
                neuron.step(sum(total_activity) * self.global_inh)
                potentials[neuron.id].append(neuron.V)

            total_activity.append(total_active_neurons)
            if len(total_activity) > 10:
                total_activity.pop(0)


        self.process_potentials(potentials)
        # divergence = self.kl_divergence(start_1=0, 
        #                                 start_2=self.time//3*2, 
        #                                 lenght=self.time//3,
        #                                 fit_von_mises=True)
        divergence = 0

        if self.plot:
            self.plot_potentials(divergence)

        return divergence 

    def process_potentials(self, potentials):
        data = pd.DataFrame(potentials)
        data.index = [self.neurons[i].angle for i in data.index]
        data.index = data.index.astype(int)

        spikes = data == 0.0
        spikes = spikes.astype(int)
        spikes = spikes.apply(lambda x: x * x.index)
        spikes = spikes.replace(0, np.nan)

        self.raw_data = data.copy()
        self.spikes = spikes.copy()

        self.flushed = False

    def kl_divergence(self, start_1, start_2, lenght, fit_von_mises=True):
        """
        Compute kl divergence between two slices of data

        Parameters
        -----------

        start_1 : where to start for the first slice
        start_2 : where to start for the second slice
        lenght : how long the slices should be
        fit_von_mises : whether to fit the slices of data to a von mises distribution

        """

        slice_1 = self.spikes.iloc[:, start_1:start_1+lenght].values.flatten()
        slice_1 = slice_1[~np.isnan(slice_1)]

        slice_2 = self.spikes.iloc[:, start_2:start_2+lenght].values.flatten()
        slice_2 = slice_2[~np.isnan(slice_2)]

        if slice_1.size > slice_2.size:
            slice_1 = slice_1[:slice_2.size]
        elif slice_2.size > slice_1.size:
            slice_2 = slice_2[:slice_1.size]

        if fit_von_mises:
            slice_1 = vonmises.fit(slice_1, fscale=slice_1.std())
            slice_2 = vonmises.fit(slice_2, fscale=slice_2.std())

            slice_1 = vonmises.rvs(*slice_1, size=100000)
            slice_2 = vonmises.rvs(*slice_2, size=100000)

        divergence = entropy(slice_1, slice_2)

        return divergence


    def input_source(self, starting_point, n_of_spikes, begin, duration, neuron, time):
        start = (starting_point - self.stim_width // 2) % self.n
        end = (starting_point + self.stim_width // 2) % self.n
        sources = [i for i in range(start, end)]
        if time > begin:
            if neuron.id in sources:
                for _ in range(n_of_spikes):
                    neuron.exc_ps_td.append(
                        ((time - begin) * 1e-3, self.weights[0]))
                    

    def flush(self,neurons=True,fixed_points=True,connections=True):
        "Reset the model so simulations can be re-run without carrying activity over"

        if neurons:
            self.neurons = [LIF(ID=i, angle=360.0/n*i, noise_mean=0, noise_std=self.noise,) for i in range(n)]
        if fixed_points:
            self.fixed_points=self.get_fixed_points()
        if connections:
            self.connect_with_fixed_points()
        self.flushed = True
        

    def connect_with_fixed_points(self):
        for neur in self.neurons:
            # if neur.id in self.fixed_points:
            #     pass
                # for i in [0] + list(range(self.n_exc_syn+1, self.n_exc_syn+self.n_inh_syn)):
                #     neur.synapses["inh"][self.neurons[(
                #         neur.id + i) % self.n]] = self.weights[3]
                #     neur.synapses["inh"][self.neurons[neur.id - i]
                #                         ] = self.weights[3]
                # for i in range(1, self.n_exc_syn+1):
                #     neur.synapses["exc"][self.neurons[(
                #         neur.id + i) % self.n]] = self.weights[2]
                #     neur.synapses["exc"][self.neurons[neur.id - i]
                #                         ] = self.weights[2]
            # else:
            if not self.is_global:
                for i in range(self.n_exc_syn+1, self.n_exc_syn+self.n_inh_syn):
                    print("self.weights[1]", self.weights[1])
                    
                    neur.synapses["inh"][self.neurons[(
                        neur.id + i) % self.n]] = self.weights[1]
                    neur.synapses["inh"][self.neurons[neur.id - i]
                                        ] = self.weights[1]
                for i in range(1, self.n_exc_syn+1):
                    neur.synapses["exc"][self.neurons[(
                        neur.id + i) % self.n]] = self.weights[0]
                    neur.synapses["exc"][self.neurons[neur.id - i]
                                        ] = self.weights[0]
            else:
                w = self.starting_weight
                for i in range(1, self.n // 2):
                    x1 = (neur.id + i) % self.n
                    x2 = (neur.id - i) % self.n

                    if w > 0:
                        neur.synapses['exc'][self.neurons[x1]] = w
                        neur.synapses['exc'][self.neurons[x2]] = w
                    elif w < 0:
                        neur.synapses['inh'][self.neurons[x1]] = abs(w)
                        neur.synapses['inh'][self.neurons[x2]] = abs(w)
                    
                    if abs(w) < self.max_weight:
                        w -= self.weight_decay

    def get_fixed_points(self):
        if self.fp_n == 0:
            return []

        if self.fp_n == 1:
            return [x for x in range(self.fp_width)]

        index = np.arange(self.n)
        interval = self.n // self.fp_n

        dist = index % interval
        low = interval // 2 - self.fp_width // 2
        high = interval // 2 + self.fp_width // 2
        
        return index[(dist >= low) & (dist <= high)]


    def plot_potentials(self, err):
        _, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(self.raw_data, vmin=-0.08, vmax=0.0, cmap="viridis", xticklabels=int(self.time/10),
                    yticklabels=12, cbar_kws={'label': "Membrane Potential (V)"}, ax=ax)


        for target in np.arange(0,len(self.fixed_points),self.fp_width):
            cur_fixed_point = np.mean(self.fixed_points[target:(target+self.fp_width)])
            plt.plot([0,self.time],[cur_fixed_point,cur_fixed_point],color='k')

        plt.xlabel("Time (ms)")
        plt.ylabel("Orientation of neuron (degrees)")
        plt.subplots_adjust(left=0.07, bottom=0.07, right=0.97, top=0.88)

        ax.set_title("Number of fixed points: {}\nNoise: {:.3e}\nWeights: {}\nDivergence: {:.6e}\nRandom seed: {}".format(
            self.fp_n, self.noise, self.weights, err, self.random_seed))

        print("date", datetime.now().strftime('%d-%m-%Y, %H:%M:%S'))
        # print("file", f"images/{datetime.now().strftime('%d-%m-%Y, %H:%M:%S')}.png")
        
        
        plt.savefig(
            f"images/{datetime.now().strftime('%d-%m-%Y, %H:%M:%S')}.png")
        plt.show()



if __name__ == "__main__":

    params = {
        'original': {'n': 100,
                    'noise': 2.0e-3,
                    'weights': (0.050, 0.100, 0.050, 0.250),
                    'fixed_points_number': 0,
                    'time': 500,
                    'plot': True,
                    'random_seed': 42,
                    'n_exc_syn': 4,
                    'n_inh_syn': 7,
                    'opto_weight': 0
                    },
        
        # 'global flow': { 'n': 100,
        #             'noise': 2.0e-3,
        #             'weights': (0.030, 0.100, 69, 69),
        #             'fixed_points_number': 0,
        #             'time': 500,
        #             'plot': True,
        #             'random_seed': 42,
        #             'n_exc_syn': 5,
        #             'n_inh_syn': 20,
        #             'opto_weight': 100,
        #             'opto_stim_begin': 200,
        #             'opto_starting_point': 55,
        #             'stim_width': 10,
        #             'is_global': True,
        #             'starting_weight': 0.040,
        #             'weight_decay': 0.005,
        #             'max_weight': 0.100
        #             },

        # 'global jump': { 'n': 100,
        #             'noise': 2.0e-3,
        #             'weights': (0.030, 0.100, 69, 69),
        #             'fixed_points_number': 0,
        #             'time': 500,
        #             'plot': True,
        #             'random_seed': 42,
        #             'n_exc_syn': 5,
        #             'n_inh_syn': 20,
        #             'opto_weight': 100,
        #             'opto_stim_begin': 200,
        #             'opto_starting_point': 70,
        #             'stim_width': 20,
        #             'is_global': True,
        #             'starting_weight': 0.042,
        #             'weight_decay': 0.005,
        #             'max_weight': 0.050
        #             },
    
        # 'global jump': { 'n': 100,
        #             'noise': 2.0e-3,
        #             'weights': (0.030, 0.100, 69, 69),
        #             'fixed_points_number': 0,
        #             'time': 500,
        #             'plot': True,
        #             'random_seed': 42,
        #             'n_exc_syn': 5,
        #             'n_inh_syn': 20,
        #             'opto_weight': 100,
        #             'opto_stim_begin': 200,
        #             'opto_starting_point': 70,
        #             'stim_width': 20,
        #             'is_global': True,
        #             'starting_weight': 0.042,
        #             'weight_decay': 0.005,
        #             'max_weight': 0.050
        #             },

        'global': { 'n': 100,
                    'noise': 2.0e-3,
                    'weights': (0.030, 0.100, 69, 69),
                    'fixed_points_number': 0,
                    'time': 500,
                    'plot': True,
                    'random_seed': 42,
                    'n_exc_syn': 5,
                    'n_inh_syn': 20,
                    'opto_weight': 100,
                    'opto_stim_begin': 200,
                    'opto_starting_point': 70,
                    'stim_width': 20,
                    'is_global': True,
                    'starting_weight': 0.042,
                    'weight_decay': 0.005,
                    'max_weight': 0.050
                    },

        # 'local jump': {  'n': 100,
        #             'noise': 2.0e-3,
        #             'weights': (0.065, 0.100, 69, 69),
        #             'fixed_points_number': 0,
        #             'time': 300,
        #             'plot': True,
        #             'random_seed': 42,
        #             'n_exc_syn': 3,
        #             'n_inh_syn': 0,
        #             'global_inh': 1.0e-9,
        #             'opto_weight': 5,
        #             'stim_width': 10
        #             },
        
        # 'local flow?': {  'n': 100,
        #             'noise': 2.0e-3,
        #             'weights': (0.065, 0.100, 69, 69),
        #             'fixed_points_number': 0,
        #             'time': 300,
        #             'plot': True,
        #             'random_seed': 42,
        #             'n_exc_syn': 6,
        #             'n_inh_syn': 0,
        #             'opto_starting_point': 45,
        #             'global_inh': 1.0e-9,
        #             'opto_weight': 10,
        #             'stim_width': 5
        #             },

        'local flow?': {  'n': 100,
                    'noise': 2.0e-3,
                    'weights': (0.065, 0.100, 69, 69),
                    'fixed_points_number': 0,
                    'time': 300,
                    'plot': True,
                    'random_seed': 42,
                    'n_exc_syn': 6,
                    'n_inh_syn': 0,
                    'opto_starting_point': 45,
                    'global_inh': 1.0e-9,
                    'opto_weight': 10,
                    'stim_width': 5
                    },

        
    }

    ring = RingAttractor(**params['local'])
    # ring = RingAttractor(**params['global'])

    error = ring.simulate()