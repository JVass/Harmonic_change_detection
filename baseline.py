import librosa as lr
from astropy.convolution import convolve, Gaussian1DKernel
import numpy as np

data, fs = lr.load("audio_files/test.wav", sr = 11025)

fmin = 110
fmax = 3520

def get_peaks_hcdf(hcdf_function, c, threshold, rate_centroids_second, centroids):
    changes = [0]
    centroid_changes = [[centroids[j][0] for j in range(0, c.shape[0])]]
    last = 0
    for i in range(2, hcdf_function.shape[0] - 1):
        if hcdf_function[i - 1] < hcdf_function[i] and hcdf_function[i + 1] < hcdf_function[i]:
            centroid_changes.append([np.median(centroids[j][last + 1:i - 1]) for j in range(0, c.shape[0])])
            changes.append(i / rate_centroids_second)
            last = i
    return np.array(changes), centroid_changes


# the repository also uses blackmanharris window, why?

tonnetz = lr.feature.tonnetz(y = data, 
                                sr = fs, 
                                n_chroma = 12,
                                n_octaves = 5,
                                bins_per_octave = 36,
                                hop_length = 1378,
                                fmin = 110)

kernel = Gaussian1DKernel(8)
smoothed_tonnetz = convolve(tonnetz, kernel)

peaks = get_peaks_hcdf()

print("Finished")