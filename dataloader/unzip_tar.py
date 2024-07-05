import os
import sys
from multiprocessing import Pool
from tqdm import tqdm
from glob import glob
import time


path = '/braindat/lab/chenyd/DATASET/MSD'
data_dir = sorted(glob(os.path.join(path, '*tar')))
def unzip_tar(data_dir):
    os.system(f'tar -xvf {data_dir} -C {path}')
    os.system(f'rm {data_dir}')
    print(f'{data_dir} is done')

if __name__ == '__main__':
    t0 = time.time()
    pool = Pool(8)
    for _ in tqdm(pool.imap_unordered(unzip_tar, data_dir), total=len(data_dir)):
        pass
    pool.close()
    pool.join()
    t1 = time.time()
    print(f'All done, time cost: {t1-t0}')