import numpy as np


DATASET_PATH = "C:\\Users\\shanm\\Downloads\\2011_09_26_drive_0001_extract\\2011_09_26\\2011_09_26_drive_0001_extract\\image_00\\data"


ORB_NFEATURES = 2000


FX = 718.856
FY = 718.856
CX = 607.1928
CY = 185.2157

K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])