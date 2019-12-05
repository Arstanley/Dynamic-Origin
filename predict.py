from models import Solver, Generator, Dynamic_generator, DataLoader
import logging
from glob import glob
import seaborn as sns
import scipy
from tqdm import tqdm
import numpy as np
from scipy.stats import entropy
import pickle
import argparse
import math
import os

DATA_PATH="/afs/crc.nd.edu/group/dmsquare/vol3/tzhao2/weibo/data/repost/distr/"
TARGET_PATH="/afs/crc.nd.edu/group/dmsquare/vol3/tzhao2/weibo/data/repost/splitted_2-50k_num/splitted_edgelist_50_merged"

TARGET_TIME=1.2

def config_logging():
    logging.basicConfig(
            level = logging.INFO,
            format = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s",
            handlers = [
                logging.StreamHandler()
                ]
            )
    return logging.getLogger()

if __name__ == "__main__":

    os.system('rm /afs/crc.nd.edu/group/dmsquare/vol3/tzhao2/weibo/data/repost/splitted_2-50k_num/splitted_edgelist_50_merged')
    os.system('cp /afs/crc.nd.edu/group/dmsquare/vol3/tzhao2/weibo/data/repost/splitted_2-50k_num/splitted_edgelist_50 /afs/crc.nd.edu/group/dmsquare/vol3/tzhao2/weibo/data/repost/splitted_2-50k_num/splitted_edgelist_50_merged')

    logger = config_logging()
    loader = DataLoader(DATA_PATH)

    cur_id = '1000000'

    with open('./uid_set.txt','rb') as file:
        uid_set = pickle.load(file)

    total_post = 0
    total_predicted_post = 0

    for idx in tqdm(range(loader.n)):
        real_data, pois, pid = loader.get_data(idx)
        s = Solver(real_data)

        total_post += np.sum(real_data)
        res = s.solve_1()

        if(len(pois) == 0):
            continue

        d_generator = Dynamic_generator(res, real_data, pois, TARGET_TIME)
        predicted = d_generator.generateSample()

        print(f'real: {np.sum(real_data)}, predict: {np.sum([math.ceil(item) for item in predicted])}')

        total_predicted_post += np.sum([math.ceil(item) for item in predicted])

        f = open(TARGET_PATH, 'a')

        for i in predicted:
            while(cur_id in uid_set):
                cur_id = str(int(cur_id)+1)
            for j in range (math.ceil(i)):
                f.write(f'{cur_id},{pid},-1\n')
            cur_id = str(int(cur_id)+1)

        f.close()

    logger.info('--Success--')

    print(f"Generated {total_predicted_post} data from {total_post} posts")

