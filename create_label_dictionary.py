import pickle
from tqdm import tqdm
from collections import defaultdict

DATA_PATH = "/afs/crc.nd.edu/group/dmsquare/vol3/tzhao2/weibo/data/repost/label-nov11.txt"



if __name__ == "__main__":

    label_dict = defaultdict(int)
    data_original = open(DATA_PATH, 'r')

    for obj in data_original:
        uid, y = obj.split(',')[0], obj.split(',')[1]
        label_dict[uid] = y

    data_original.close()

    print(len(label_dict))

    ### Pickle the dictionary ###
    f = open('label_dict.txt', 'wb')
    pickle.dump(label_dict, f)
    f.close()

    print("Label Dictionary Created Successfully")


