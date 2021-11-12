import numpy as np
import glob
import random

def split_data(dataset_size, test_p=0.95):
    num_train = int(dataset_size * test_p)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    return indices[:num_train], indices[num_train:]

#n = 6000
observations = None
env = None
obj = None

#all_files = glob.glob("/global/scratch/users/patrickhaoy/s3doodad/affordances/combined_new/*_images.npy") #glob.glob("/2tb/home/patrickhaoy/data/affordances/combined_new/*_images.npy")
#all_files = glob.glob("/global/scratch/users/patrickhaoy/s3doodad/affordances/combined_reset_free_v5/*_images.npy") 
all_files = glob.glob("/2tb/home/patrickhaoy/data/affordances/data/antialias_reset_free_v5_rotated_top_drawer/*_images.npy")
random.shuffle(all_files)

for filename in all_files:
    print(filename)
    try:
        data = np.load(filename, allow_pickle=True)
    except:
        print("COULDNT LOAD ABOVE FILE")
        continue
    
    data = data[()]
    if observations is None:
        observations = data['observations']
        env = data['env']
        obj = data['object']
    else:
        observations = np.concatenate((observations, data['observations']), axis=0)
        env = np.concatenate((env, data['env']), axis=0)
        obj.extend(data['object'])

# SAVE IMAGES FOR REPRESENTATION TRAINING #
#folder = '/global/scratch/users/patrickhaoy/s3doodad/affordances/combined_new/' #'/2tb/home/patrickhaoy/data/affordances/combined_new/' #'/media/ashvin/data2/data/uniform_data/'
#folder = '/global/scratch/users/patrickhaoy/s3doodad/affordances/combined_reset_free_v5/'
folder = "/2tb/home/patrickhaoy/data/affordances/data/antialias_reset_free_v5_rotated_top_drawer/"

train_i, test_i = split_data(observations.shape[0])

train = {'observations': observations[train_i], 'env': env[train_i], 'object': [obj[i] for i in train_i]}
test = {'observations': observations[test_i], 'env': env[test_i], 'object': [obj[i] for i in test_i]}

np.save(folder + 'combined_images.npy', train)
np.save(folder + 'combined_test_images.npy', test)