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
    # This is the same as the baseline. They are needed for plotting.
    tonnetz = lr.feature.tonnetz(y = data, 
                                    sr = fs, 
                                    n_chroma = number_of_chroma,
                                    n_octaves = number_of_octaves,
                                    bins_per_octave = bins_per_octave,
                                    hop_length = hop_length,
                                    fmin = fmin,
                                    norm = 1)
    filter_kernel = Gaussian1DKernel(std)
    smoothed_tonnetz = convolve_per_feature(tonnetz, filter_kernel)
    hcdf = harmonic_change_function(tonnetz, euclidean_distance)
    smoothed_hcdf = harmonic_change_function(smoothed_tonnetz, euclidean_distance)
    anno = annotations()
    chord_changes = anno.fetch_changes(song_name)
    test = peak_finder_k_neighbours(smoothed_hcdf,peak_estimation_filter_size)
    indexes_of_peaks = np.where(test==1)
    indexes_of_peaks = indexes_of_peaks[1]

    # Plotting the Tonnal Centroid Calculation and its Smoothed version
    plt.subplot(2,1,1)
    plt.imshow(cv.resize(tonnetz[:, 200:400], dsize = (200,20)), label = "Tonnetz")
    plt.subplot(2,1,2)
    plt.imshow(cv.resize(smoothed_tonnetz[:,200:400], dsize = (200,20)), label = "Smoothed version of Tonnetz")
    plt.legend()
    plt.savefig("./output/tonnetz_and_smoothed_tonnetz.png")
    plt.close()

    # Plotting the HCDF and the Smoothed HCDF version
    plt.plot(np.squeeze(hcdf), label = "HCDF")
    plt.plot(np.squeeze(smoothed_hcdf), label = "Smoothed HCDF")
    plt.legend()
    plt.savefig("./output/hcdf_and_smoothed_hcdf.png")
    plt.close()

    # Plotting the peaks that were estimated for the smoothed hcdf, the ones that were a hit and the ones that were not and the transcription times as vertical lines
    plt.figure(figsize = (50,10))
    plt.plot(np.squeeze(smoothed_hcdf), label = "Smoothed HCDF")
    plt.plot(indexes_of_peaks, smoothed_hcdf[0,indexes_of_peaks], 'x', label = "Peaks estimated")

    _, used_peak_indexes, unused_peak_indexes = calculate_hits(indexes_of_estimated_peaks = indexes_of_peaks, chord_changes = chord_changes, chroma_parameters = chroma_parameters)

    plt.scatter(used_peak_indexes, smoothed_hcdf[0,used_peak_indexes], label = "Hits", edgecolors = "green", facecolors = "none")
    plt.scatter(unused_peak_indexes, smoothed_hcdf[0, unused_peak_indexes], label = "No-Hits", edgecolors = "red", facecolors = "none")

    # Convert the timestamps to frame indexes
    timestamps_per_frame = np.arange(0, smoothed_hcdf.shape[1], 93e-3)
    for change in chord_changes:
        frame_index_for_timestamp = np.where(np.logical_and(change <= timestamps_per_frame + 93e-3, timestamps_per_frame <= change))
        plt.axvline(x = frame_index_for_timestamp, color = (167/255, 196/255, 242/255), label = 'axvline - full height')

    plt.savefig("./output/smoothed_hcdf_peaks_and_hits.png")
    plt.close()