from scipy.io import loadmat

data = "data/seed_iv/eeg_raw_data/1/2_20150915.mat"
annots = loadmat(data)
print(annots)
