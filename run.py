from models import Solver, Generator, dynamic_generator
import logging
from glob import glob
import seaborn as sns
import scipy
from tqdm import tqdm
import numpy as np
from scipy.stats import entropy
import pickle

DATA_PATH="/afs/crc.nd.edu/group/dmsquare/vol3/tzhao2/weibo/data/repost/distr/"

def min_max(d):
    return np.min(d), np.max(d)

def scale(_min, _max, arr):

    return np.interp(arr, (min(arr), max(arr)), (_min, _max))

def customKL(d_1, d_2):
    ### Epsilon to avoid infinte kl ###
    epsilon = 0.0001
    p = [epsilon] * 100
    q = [epsilon] * 100
    n = len(d_1)

    d_1, d_2 = scale(0, 100, d_1), scale(0, 100, d_2)

    p = [num / sum(d_1) for num in d_1]
    q = [num / sum(d_2) for num in d_2]

    return entropy(p, q)

with open('./label_dict.txt','rb') as file:
    label_dict = pickle.load(file)

class DataLoader:
    def __init__(self, path):
        # Get the path to data
        self.path = path
        self.files = [f for f in glob(DATA_PATH+"*")]
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


def config_logging():
    logging.basicConfig(
            level = logging.INFO,
            format = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s",
            handlers = [
                logging.StreamHandler()
                ]
            )
    return logging.getLogger()

def generate(parameters, sample_size):
    g = Generator(parameters)
    _generated = g.generateSample(sample_size)
    return _generated

def compare_kl(original_data, generated_a, generated_b):
    kl1, kl2 = customKL(original_data, generated_a), customKL(original_data, generated_b)
    return (generated_b, kl2) if 0 == 1 else (generated_a, kl1)

def compute_best_model(data):
    s = Solver(data)
    res_1 = s.solve_1()
    res_not_1 = s.solve_not_1()

    _generated_1 = generate(res_1, len(data))
    _generated_n1 = generate(res_not_1, len(data))

    return compare_kl(data, _generated_1, _generated_n1)

def save_graph(dist_a, dist_b, kl, pid, pos_or_neg):
    graph = sns.distplot(dist_a)
    sns.distplot(dist_b)
    graph.set_title(f"pid: {pid}, kl: {kl}, pos/neg: {pos_or_neg}")
    graph.get_figure().savefig(f"./Imgs/{pid}_{pos_or_neg}.png")
    graph.get_figure().clf()

if __name__ == "__main__":

    logger = config_logging()
    logger.info("--Program Starts --")

    ### Get the list of pid ###
    loader = DataLoader(DATA_PATH)

    ### Fitting model to data ###
    logger.info("---- Fitting the model to empirical data ----")

    for i in tqdm(range(loader.n)):
        pos_u, pid = loader.get_data_pos(i)
        neg_u = loader.get_data_neg(i)[0]

        generated_pos, kl_pos = compute_best_model(pos_u)
        generated_neg, kl_neg = compute_best_model(neg_u)

        save_graph(generated_pos, pos_u, kl_pos, pid, 'T')
        save_graph(generated_neg, neg_u, kl_neg, pid, 'F')

    logger.info("--Success--")
