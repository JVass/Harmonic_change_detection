import numpy as np

hp = np.array([0.53, 0.72, 0.5, 0.59, 0.45, 0.53, 0.49, 0.49, 0.29, 0.64, 0.49, 0.53, 0.34, 0.65, 0.5, 0.68])
Hn = np.array([128, 126, 129, 142, 158, 133, 169, 133, 152, 132,148,160,128,127,213,160])

hits_of_precission = hp * Hn

hr = np.array([0.87,0.8,0.86,0.9,0.7,0.87,0.82,0.67,0.74,0.86,0.86,0.9,0.81,0.83,0.88,0.95])
TC = np.array([78,113,74,93,101,81,101,96,59,97,84,94,54,98,120,113])

hits_of_recall = hr * TC

print("Hits of precision and recall respectively")
print(np.round(hits_of_precission))
print(np.round(hits_of_recall))

# Hits per song with Harte and Sandler algorithm
# 68.  91.  64.  84.  71.  70.  83.  65.  44.  84.  73.  85.  44.  83.  106.  109