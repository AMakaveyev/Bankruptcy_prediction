
import pandas as pd
import numpy as np
from scipy.io import arff

dataset = arff.loadarff(open(r'/Users/anatoliymakaveyev/Documents/Python Environment/BANKRUPTCY_PREDICTION_CASE/data/1year.arff'))
df_data = pd.DataFrame(dataset[0])