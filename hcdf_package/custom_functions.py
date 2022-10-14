import librosa as lr
from astropy.convolution import convolve, Gaussian1DKernel
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import csv
from scipy import signal as sg
import cv2 as cv


class annotations():
    """
    To Do
    Add the frequency sampling rate as a parameter
    """
    def __init__(self, path_to_annotations = "./annotations/"):
        self.song_names = os.listdir(path_to_annotations)

        self.change = {}

        for song in self.song_names:
            changes = []

            with open(path_to_annotations + song, "r") as file:
                csv_reader = csv.reader(file, delimiter = " ")
                for row in csv_reader:
                    if len(row) < 3:
                        continue
                    changes.append(float(row[1]))
        
            self.change[song] = np.array(changes)

    def fetch_changes(self, name_of_song):
        return self.change[name_of_song + ".csv"][:-1]

    def fetch_changes_from_time_to_frame(self, name_of_song, window_size, hop_length):
        self.change_frame_domain = self.change[name_of_song + ".csv"] * fs

def euclidean_distance(x,y):
    """
    A wrapper for the euclidean distance to be passed as an argument to the harmonic change function

    Input:
        x (numpy array (N, number_of_frames)): N is the number of features and number_of_frames is the number of frames of the 
                                                chromagram
        y: (numpy array (N, number_of_frames)): N is the number of features and number_of_frames is the number of frames of the 
                                                chromagram

    Output:
        The euclidean distance between x and y                                            
    """
    return np.linalg.norm(x-y,2)

# the repository also uses blackmanharris window, why?

def harmonic_change_function(chromagram, similarity_measure):
    """
    Input:
        chromagram (numpy array, (N,number_of_frames)): This is matrix representation of the audio input. 
                                                        It can be a chromagram or any transformation of a chromagram.
                                                        E.g. if it is a chromagram, N will be equal to the number of chroma 
                                                        of the chromagram. If it is a Tonnetz, N will be equals 6.
        similarity_measure (function): A metric of similarity between the n+1 and n-1 frame. On the paper of Harte and Sandler is a
                                        Euclidean distance but it can be every available metric.                                             
    Output:
        HCDF (1, number_of_frames)): The harmonic change detection function, which means the similarity of n+1, n-1 frame w.r.t. the 
                                    similarity measure that was chosen. 

    Comments:
        In the paper of Harte and Sandler it is explicitly written that
        "hcdf is defined as the rate of change of the smoothed tonal centroid vectors". So, it has to be devided
        by the number of centroids per frame?
    """

    _, number_of_frames = chromagram.shape
    HCDF = np.zeros(shape = (1,number_of_frames))

    for frame in range(1, number_of_frames-1):
        HCDF[0, frame] = similarity_measure(chromagram[:,frame+1], chromagram[:,frame-1])

    return HCDF

def convolve_per_feature(chromagram, kernel):
    """
    A function to convole each row (feature) of the chromagram with a kernel (filter).

    Input:
        chromagram (numpy array, (N, number_of_frames)): This is matrix representation of the audio input. 
                                                        It can be a chromagram or any transformation of a chromagram.
                                                        E.g. if it is a chromagram, N will be equal to the number of chroma 
                                                        of the chromagram. If it is a Tonnetz, N will be equals 6.
        kernel (astropy kernel): The kernel that will be used for the convolution                                                        
    Output:
        row_convolved_chromagram (numpy array, (N, number_of_frames)): The row-convolved chromagram matrix
    """
    N, number_of_frames = chromagram.shape
    row_convoled_chromagram = np.zeros(shape = (1, number_of_frames))

    for row_index in range(N):
        row_convoled_chromagram = np.vstack((row_convoled_chromagram, convolve(chromagram[row_index, :], kernel, normalize_kernel = True)))

    row_convoled_chromagram = row_convoled_chromagram[1:, :]

    return row_convoled_chromagram

def absolute_thresholding(hcdf, threshold):
    """
    From the paper "Harmonic change detection for musical chords segmentation" form Degani et al, a threshold is chosen for peak estimation.

    This will be implemented for now, and then trying more complicated peak estimation algorithms.

    Input:
        hcdf (np array, (1,number_of_frames)): the normalized harmonic change detection function

        threshold: the threshold number for absolute thresholding

    Output:
        peak_hcdf (np array (1, number_of_frames)): the peaks that were found in 

    Comments:
        This kind of thresholding can introduce a certain region of "peaks". We have to decide the most probable candidate for them.
    """

    peak_hcdf = np.zeros_like(hcdf)
    _, number_of_frames = hcdf.shape

    for frame in range(number_of_frames):
        if hcdf[0, frame] >= threshold:
            peak_hcdf[0, frame] = 1
    
    return peak_hcdf

def peak_finder_k_neighbours(hcdf, k = 3):
    """
    A simple algorithm for finding the peak in a function
    
    Input:
        hcdf (np array, (1,number_of_frames)): the normalized harmonic change detection function
        k: the number of neighbourhood (minus k//2 and plus k//2 index with respect to the current index) to take into consideration
    
    Output:
        peak_hcdf (np array (1, number_of_frames)): the peaks that were found
    """
    peak_hcdf = np.ones_like(hcdf)
    peak_hcdf[0, :k//2] = 0
    peak_hcdf[0, -k//2:] = 0
    _, number_of_frames = hcdf.shape

    for frame in range(k//2 + 1 , number_of_frames - k//2 - 1):
        for testing_index in range(-k//2 + frame + 1, frame + k//2 + 1):
            if hcdf[0,testing_index] > hcdf[0,frame]:
                peak_hcdf[0,frame] = 0
                break

    return peak_hcdf

def Foote_metric(y_true, y_pred):
    """
    Calcualte the Footes measure between two different vectors

    Input:
        y_true (np array (N,1)): Xn vector on the Hainsworth paper
        y_pred (np array (N,1)): Xn-1 vector on the Hainsworth paper

    Output:
        The Foote's metric between y_true and y_pred
    """
    pass

def hainsworth_peak_finder(hcdf):
    # smoothing
    hanning_window = np.expand_dims(sg.windows.hann(65), axis = 0)
    smoothed_function = sg.convolve(hcdf, hanning_window)

    # finding the mean value of the unsmoothed function
    mean_unsmooth_value = np.mean(smoothed_function)

    # finding the part that is higher than the mean value
    higher_than_mean = np.where(smoothed_function >= mean_unsmooth_value, 0, smoothed_function)[0]
    higher_than_mean = np.expand_dims(higher_than_mean, axis = 0)

    # of those, find the peak
    smooth_hcdf_peaks = peak_finder_k_neighbours(higher_than_mean, 12)

    # filter them in a sequential pair-wise fashion using Foote's measure

    return smooth_hcdf_peaks 

def effective_frame_to_time(frame_no, chroma_parameters):
    lower_thr = 0
    higher_thr = 0

    # Each frame starts from (frame_no-1) * hop_sample and ends at (frame_no-1)*hop + window_length sample, of window_length total duration
    # Also, a hit is defined as the +- frames with respect to the center frame. Therefore, the effective window starts from
    # (frame_no-1 - 3) * hop_sample and ends at (frame_no-1 + 3) + window_length
    lower_thr = (frame_no-1-3) * chroma_parameters["hop"] if (frame_no-1-3) * chroma_parameters["hop"] > 0 else 0
    higher_thr = (frame_no -1 + 3) * chroma_parameters["hop"] + chroma_parameters["window"]

    lower_thr = lower_thr * (1 / chroma_parameters["fs"])
    higher_thr = higher_thr * (1 / chroma_parameters["fs"])

    return (lower_thr, higher_thr)

def calculate_hits(indexes_of_estimated_peaks, chord_changes, chroma_parameters):
    # What about two consecutive hits?
    hits = 0

    indexes = copy.deepcopy(indexes_of_estimated_peaks)
    used_indexes = []

    for time_of_chord_change in chord_changes:
        temp_frame_to_test = 0
        remaining_indexex_to_test = len(indexes)

        while(temp_frame_to_test < remaining_indexex_to_test or remaining_indexex_to_test == 0):
            lower_time, higher_time = effective_frame_to_time(indexes[temp_frame_to_test], chroma_parameters = chroma_parameters)

            if time_of_chord_change >= lower_time and time_of_chord_change <= higher_time:
                hits += 1
                used_indexes.append(indexes[temp_frame_to_test])
                indexes = np.delete(indexes, temp_frame_to_test)  

                break

            temp_frame_to_test += 1

    return hits, np.array(used_indexes), np.array(indexes)
