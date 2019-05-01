import pandas as pd
import numpy as np
def load_data(file):
	data = np.array(pd.read_csv(file,sep=",",header = None))
	real_data = data[:,1:-1]
	label = data[:,-1]
	return real_data, label
x,y = load_data("glass.data")
