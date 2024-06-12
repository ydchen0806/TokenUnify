import h5py
f_raw = h5py.File('/h3cstore_ns/Backbones/data/wafer/wafer36_inputs.h5', 'r')
data = f_raw['main'][:]
print(data.shape)
f_raw.close()

# data
# import thop
