import os
import numpy as np


root_dir = '/home/ps/jinfei/workspace/self-critical.pytorch-master/data/news_annotations/news_2_features_npy/'
output_dir = '/home/ps/jinfei/workspace/self-critical.pytorch-master/data/newsbu'


#os.makedirs(output_dir+'_att')
os.makedirs(output_dir+'_spatial')

split_list = os.listdir(root_dir) # [train2017, valid2017]

for split in split_list:
    split_path = os.path.join(root_dir, split)
    for file_npy in os.listdir(split_path):
        data = np.load(os.path.join(split_path,file_npy))
        np.save(os.path.join(output_dir+'_spatial', file_npy), data)
        #np.save(os.path.join(output_dir+'_att', file_npy.split('.')[0]), data)




