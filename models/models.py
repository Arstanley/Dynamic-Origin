import numpy as np
from glob import glob
from numpy import log
from scipy.optimize import minimize, fsolve
from matplotlib import pyplot as plt
import seaborn as sns
import scipy

DATA_PATH="/afs/crc.nd.edu/group/dmsquare/vol3/tzhao2/weibo/data/repost/distr/"

def min_max(arr):
    arr_max = max(arr)
    arr_min = min(arr)
    new_arr = [(item-arr_min)/(arr_max-arr_min)+0.000001 for item in arr]
    return new_arr

class DataLoader:
    def __init__(self, path):
        # Get the path to data
        self.path = path
        self.files = [f for f in glob(path+"*")]
        self.n = len(self.files)

    def get_file_list(self):
        return self.files

    def get_data_neg(self, idx):
        f = open(self.files[idx])
        data = [int(line.split(',')[1]) for line in f if label_dict[line.split(',')[0]] == "F\n"]
        f.close()
        return data, self.files[idx].split('/')[-1]

    def get_data_pos(self, idx):
        f = open(self.files[idx])
        data = [int(line.split(',')[1]) for line in f if label_dict[line.split(',')[0]] == "T\n"]
        f.close()
        return data, self.files[idx].split('/')[-1]

    def get_data(self, idx):
        f = open(self.files[idx])
        data, pois = [], []
        for line in f:
            line = line.split(',')
            if (len(line) == 3):
                data.append(int(line[1]))
                pois.append(int(line[2]))

        if pois == []: print(self.files[idx].split(
            '/'
            ))
        pois = min_max(pois)

        f.close()
        # Return data, enterring time, and post id
        return data, pois, self.files[idx].split('/')[-1]

class Solver:
    def __init__(self, data): # data: 1-d array like input
        self.func_not_1 = lambda params: -(np.sum(np.log(params[0] + params[1] * np.power(data + params[2], -params[3])))
                                          - params[0] * np.sum(data) - (params[1]/(1-params[3])) * np.sum(np.power(data+params[2], 1-params[3]) - np.power(params[2], 1-params[3])))
        self.func_1 = lambda params: -(np.sum(np.log(params[0] + params[1] * np.power(data + params[2],-1))) - params[0] * np.sum(data) -
                                      params[1] * np.sum(np.log((data / params[2]) + 1)))
        self.bnds = ((0, None),(0, None),(1, None),(0, None))
    def solve_1(self):
        return minimize(self.func_1, (1, 1, 1, 1), method='SLSQP', bounds=self.bnds)
    def solve_not_1(self):
        return minimize(self.func_not_1, (0.0005, 1, 5, 0), method='SLSQP', bounds = self.bnds)


class Generator:
    def __init__(self, params):
        self.beta = params.x[0]
        self.alpha = params.x[1]
        self.Lambda = params.x[2]
        self.Theta = params.x[3]
    def generateSample(self, n):
        return [self.newton_iter_solver(np.random.rand()) for _ in range (n)]
    def newton_iter_solver(self, u):
        threshold = 10**(-8)
        x = 0.
        phi = 0
        i = 1
        while (abs(phi) < threshold):
            phi = np.log(u) + self.beta * x + self.alpha * np.log((x/self.Lambda) + 1)
            dphi = self.beta + self.alpha * (self.Theta / (x + self.Lambda))
            x -= (phi/dphi)
            i += 1
        return x


class Dynamic_generator:
    # Params: Trained Parameters, x: current state x in vector, pois: Time point that x_i enter the system, t: time point that we want to predict
    def __init__(self, params, x, pois, t):
        self.beta = params.x[0]
        self.alpha = params.x[1]
        self.Lambda = params.x[2]
        self.Theta = params.x[3]
        self.pois = pois
        self.x = x
        self.target_t = t

    def generateSample(self):
        return [self.newton_iter_solver(i) for i in range (len(self.x))]

    def newton_iter_solver(self, index):
         threshold = 10**(-4)
         x = 1
         phi = 0.000001
         while (abs(phi) < threshold):
             phi = self.beta * x + self.alpha * np.log((x/self.Lambda) + 1) - np.log(self.target_t/self.pois[index])
             dphi = self.beta + self.alpha * (1 / (x/self.Lambda+1)) * (1 / self.Lambda)
             x -= (phi/dphi)
         return x

        # threshold = 10**(-4)
        # x = self.x_0
        # i = 1
        # lr = 0.05
        # thresh = 1e-4
        # phi = self.beta * x + self.alpha * np.log((x/self.Lambda) + 1) - np.log(self.process[-1]/self.process[index])
        # dphi = self.beta + self.alpha * (1 / (x / (self.Lambda)+1)) * (1 / self.Lambda)
        # while abs(phi) > thresh:
        #     phi = self.beta * x + self.alpha * np.log((x/self.Lambda) + 1) - np.log(self.process[-1]/self.process[index])
        #     dphi = self.beta + self.alpha * (1 / (x / (self.Lambda)+1)) * (1 / self.Lambda)
        #     x -= lr * dphi
        #     i += 1
        #     #print(i, dphi, phi)
        # return x


    # def newton_iter_solver(self, index):
    #     threshold = 10**(-4)
    #     x = self.x_0
    #     phi = 0
    #     i = 1
    #     while (abs(phi) < threshold):
    #         phi = self.beta * x + self.alpha * np.log((x/self.Lambda) + 1) - np.log(self.process[-1]/self.process[index])
    #         dphi = self.beta + self.alpha * (1 / (x / (self.Lambda)+1)) * (1 / self.Lambda)
    #         x -= (phi/dphi)
    #     return x
