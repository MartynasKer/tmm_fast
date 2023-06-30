import numpy as np
import tmm_fast.plotting_helper as plth
from tmm_fast.vectorized_tmm_dispersive_multistack import coh_vec_tmm_disp_mstack as tmm
import matplotlib.pyplot as plt
import os
import pandas as pd


class Environment:
    def __init__(self, stacks, wl, max_layers):
        self.init_hyperparams()

        self.theta_degrees = np.linspace(0, 90, 10)
        theta = self.theta_degrees * (np.pi / 180)

        self.theta = theta

        self.stacks = stacks
        self.max_layers = max_layers
        self.M = np.ones((stacks, self.max_layers, wl.shape[0]), dtype=np.complex128)
        self.current_layers = np.ones((self.M.shape[0]), dtype=np.int16)

        self.target_r = np.zeros((stacks, wl.shape[0]))
        self.target_t = np.zeros((stacks, wl.shape[0]))

        self.current_layers[:] = 2
        # tarp stiklo ir oro
        self.M[:, :, :] = 1 + 0.0j
        self.M[:, -1, :] = 1.8 + 0j
        self.W = np.zeros((self.M.shape[0], self.M.shape[1]), dtype=np.complex128)
        self.W[:, 0] = np.inf
        self.W[:, -1] = np.inf

        self.obs = np.zeros((stacks, self.max_layers, 2))
        self.delta_fitness = np.zeros((stacks))

        SiO2_path = "materials/preprocessed/nSiO2.txt"
        ZnO_path = "materials/preprocessed/nZnO.txt"
        TiO2_path = "materials/preprocessed/nTiO2.txt"
        self.import_advanced_layers(SiO2_path, index=0)
        self.import_advanced_layers(ZnO_path, index=1)
        self.import_advanced_layers(TiO2_path, index=2)

    def import_advanced_layers(self, layer_file_path, index=None):
        """
        Import advanced layers from a file
        """
        if not os.path.exists(layer_file_path):
            raise ValueError(f"File {layer_file_path} does not exist")

        data = pd.read_csv(layer_file_path, sep="\t", header=None)
        wl = data[0]
        n = data[1]
        adv_layer = np.empty((1, max(wl) - min(wl)), dtype=np.complex128)
        for i in range(len(wl)):
            adv_layer[:, i - 1] = n[i]
        if index is not None:
            self.layer_options[index] = adv_layer
        else:
            self.layer_options.append(adv_layer)

    def add_layer(self, layer, thickness, stack_id=0, from_options=False):
        pos = self.current_layers[stack_id] - 1
        if from_options:
            if layer >= len(self.layer_options):
                raise ValueError(f"Layer {layer} is not in the options")

            self.M[stack_id, pos] = self.layer_options[layer]
            self.obs[stack_id]
            w = np.complex128()
            w = self.thickness_options[thickness]
        else:
            w = thickness
            self.M[stack_id][pos][:] = layer
        self.W[stack_id, pos] = w
        self.current_layers[stack_id] += 1
        self.obs[stack_id, pos - 2] = np.array([layer, thickness])

    def plot_env(self):
        assert (
            self.M.shape[1] >= 3
        ), "At least 3 layers are needed to plot the environment"

        fig, ax = plt.subplots(1, 1)
        ax, cmap = plth.plot_stacks(ax, list(self.M[:, 1:-1, 0]), list(self.W[:, 1:-1]))

    def get_reflection(self, wl):
        return tmm("s", self.M, self.W, self.theta, wl, device="cpu")["R"]

    def get_transmission(self, wl):
        return tmm("s", self.M, self.W, self.theta, wl, device="cpu")["T"]

    def compute(self, wl):
        return tmm("s", self.M, self.W, self.theta, wl, device="cpu")

    def step(self, actions):
        """
        action: [layer, thickness, reset]
        layer: chosen from 3 different materials: [SiO2, ZnO, TiO2]
        thickness: chosen from 10 different thicknesses: [10, 50, 100, 150, 200, 250, 300, 350, 400, 450] nm
        reset: 1 or 0 terminates the episode
        """
        reset_flag = np.zeros((len(actions)))
        done = np.zeros((len(actions)))

        for idx, action in enumerate(actions):
            layer = action[0]
            thickness = action[1]
            # reset = action[2]
            self.add_layer(layer, thickness, idx, True)

            # if reset:
            #    reset_flag[idx] = 1
            #    done[idx] = True
        wl = np.linspace(300, 1500, 1200) * (10 ** (-9))
        computed_values = tmm(
            "s", self.M, self.W, np.array([np.pi / 4]), wl, device="cpu"
        )
        T = computed_values["T"]
        R = computed_values["R"]
        reward = np.zeros((len(T)))
        observations = self.obs
        for idx in range(len(T)):
            if reset_flag[idx] == 1:
                self.reset(idx)

                continue
            reward[idx] = self.get_reward(idx, R[idx], T[idx])
            done[idx] = False
            # print(self.current_layers[idx])
            if self.current_layers[idx] >= self.max_layers:
                done[idx] = True
                self.reset(idx)

        return observations, reward, done

    def set_targets(self, stack_id, target_t, target_r):
        self.target_r[stack_id] = target_r
        self.target_t[stack_id] = target_t

    def get_fitness(self, stack_id, r, t):
        r_error = np.mean((r - self.target_r[stack_id]) ** 2)
        t_error = np.mean((t - self.target_t[stack_id]) ** 2)

        return 1 - (r_error + t_error)

    def get_reward(self, stack_id, r, t):
        fitness = self.get_fitness(stack_id, r, t)
        reward = fitness - self.delta_fitness[stack_id]
        self.delta_fitness[stack_id] = fitness
        return reward - 0.05

    def get_actions(self):
        # layer, thickness, reset
        return [len(self.layer_options), len(self.thickness_options)]

    def get_state(self, stack_id):
        return self.obs[stack_id]

    def reset(self, stack_id):
        self.M[stack_id] = np.ones((self.max_layers, self.M.shape[2]))
        self.W[stack_id] = np.zeros((self.max_layers))
        self.M[stack_id, 0, :] = 1
        self.M[stack_id, -1, :] = 1.8
        self.W[stack_id, 0] = np.inf
        self.W[stack_id, -1] = np.inf
        self.obs[stack_id] = np.zeros((self.max_layers, 2))
        self.current_layers[stack_id] = 2
        self.delta_fitness[stack_id] = 0

    def init_hyperparams(self):
        self.max_layers = 30
        self.max_thickness = 450
        self.min_thickness = 10
        self.max_stacks = 128
        self.min_stacks = 1
        self.max_wl = 1500
        self.min_wl = 300
        self.max_theta = 90
        self.min_theta = 0
        self.max_reward = 1
        self.min_reward = 0
        self.max_episode_steps = 10
        self.max_steps = 10
        self.max_episode_reward = 10

        self.thickness_options = np.array(
            [1, 5, 10, 15, 20, 25, 30, 35, 40, 45], dtype=np.complex128
        )
        self.thickness_options *= 10 ** (-9)
        self.layer_options = np.empty((3, 1, 1200), dtype=np.complex128)
