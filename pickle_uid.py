import pickle
from tqdm import tqdm
from collections import defaultdict

DATA_PATH = "/afs/crc.nd.edu/group/dmsquare/vol3/tzhao2/weibo/data/repost/splitted_2-50k_num/splitted_edgelist_50"


if __name__ == "__main__":

    uid_set = set()
    data_original = open(DATA_PATH, 'r')

    for obj in data_original:
        uid = obj.split(',')[0]
        uid_set.add(uid)

    data_original.close()

    print(len(uid_set))

    ### Pickle the dictionary ###
    f = open('uid_set.txt', 'wb')
    pickle.dump(uid_set, f)
    f.close()

    print("User ID Set Created Successfully")


