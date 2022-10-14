from hcdf_package.custom_functions import *

# Output to reproduce
# Song   Title     TC Mn  Hp  Hr  Hf    Hits
# Please Please Me 78 128 53% 87% 65.9% 68

fmin = 110
fmax = 3520
number_of_chroma = 12
number_of_octaves = 5
bins_per_octave = 36
hop_length = 1024
window_size = 8192
fs = 11025
std = 8**(1/2)
peak_estimation_filter_size = 7 # in total samples, half-1 on the right and half-1 on the left

chroma_parameters = {
    "window": window_size,
    "hop": hop_length,
    "fs": fs
}

data, fs = lr.load("audio_files/test.wav", sr = fs)

song_name = "please_please_me"

if __name__ == "__main__":
    """
    1) Costant Q
    2) 12-bin Tuned chromagram
    3) Tonal Centroid Transform
    """
    tonnetz = lr.feature.tonnetz(y = data, 
                                    sr = fs, 
                                    n_chroma = number_of_chroma,
                                    n_octaves = number_of_octaves,
                                    bins_per_octave = bins_per_octave,
                                    hop_length = hop_length,
                                    fmin = fmin,
                                    norm = 1)

    """
    4) Smoothing function with a Gaussian 8-std kernel
    Std is inversely proportionate with the filters length. 
    The length of a chroma segment is N*hop*T = N * 93ms

    For std = 8, the filter's length is 65 samples (on chromagram)
    and therefore the duration of it is 65 * 93 = 6.045 seconds
    For std = sqrt(8), the filter's length is 23 and its duration 23 * 93 = 2.139 seconds
    """
    filter_kernel = Gaussian1DKernel(std)
    smoothed_tonnetz = convolve_per_feature(tonnetz, filter_kernel)

    """
    Distance calculation
    # """
    hcdf = harmonic_change_function(tonnetz, euclidean_distance)
    smoothed_hcdf = harmonic_change_function(smoothed_tonnetz, euclidean_distance)

    # ANNOTATIONS, this has to be implemented as a function of the object annotations
    anno = annotations()
    chord_changes = anno.fetch_changes(song_name)

    # Harte and Sandlers pipeline with custom peak finder
    test = peak_finder_k_neighbours(smoothed_hcdf,peak_estimation_filter_size)
    indexes_of_peaks = np.where(test==1)
    indexes_of_peaks = indexes_of_peaks[1]

    hits, _, _ = calculate_hits(indexes_of_peaks, chroma_parameters = chroma_parameters, chord_changes = chord_changes)

    TC = len(chord_changes)
    Mn = len(indexes_of_peaks)

    recall = hits / TC
    precision = hits / Mn
    f1_score = (2 * precision * recall) / (precision + recall)

    print(f"{song_name} score:")
    print(f"TC: {TC}, Mn: {Mn}, Number of hits: {hits}, Precision {precision}, Recall: {recall}, f1: {f1_score}")