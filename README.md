# ECS7007P-RMRI
## Done so far
The baseline/spine for our experiments are implemented. The procedure can be describe in steps:
1. Constant Q Transform calculation: it is handled by librosa.feature.tonnetz for Harte and Sandler
2. 12-bin Chromagram: it is handled by librosa.feature.tonnetz for Harte and Sandler
3. Calculation of the Tonal Centroids (TC - mapping to Tonnetz/torrus space): it is handled by librosa.feature.tonnetz for Harte and Sandler
4. Smoothing: A gaussian blurring is applied to each dimension of the 6-dimensional Tonal Centroids. The paper explicitly says that the sigma is 8.
5. Applying a distance function to estimate the rate of TC change: Euclidean distance. I have implemented a wrapper to use whatever metric we want
6. Peak finder: An algorithm for peak estimation. For this, a simple window based peak estimation has been implemented. On the baseline algorithm by Hainsworth and Macleod, the procedure is described. Maybe, we have to use that.
7. Calculate hits: A hit is defined as a peak that is +-3 frames away from a transcripted change. For this implementation, the first predicted peak is the "pair" with the transcribed chord change. 
8. Metrics calculation: Recall is defined as the ratio of hits and transcripted change. Precision is defined as the ration of hits and predicted change. F1 follows the typicall definition but based on the aforementioned Recall and Precission metrics.

## Questions:
1. What happens when a transcribed peak belongs to more than one frame? For what peak should it count?
2. What is the value of std for the Gaussian blurring?
3. What is the peak finding algorithm? In section 3.2 it says that the HCDF is "defined as the overall rate of change of the smoothed tonal centroid signal"

## To-Do
1. Implement the Hainsworth peak finder algorithm.
2. Add a function to read the song/annotations pair to automate the experiment
3. Add a function to write the results in a csv format like the one that is used on the papers for ease of comparison.
4. Introduce the NNLS Chromagram algorithm from the vamp package.
5. Implement the Hainsworth paper and reproduce its results.
