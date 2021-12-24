import matplotlib.pyplot as plt
import pandas as pd 

data = pd.read_csv("scores.csv")

plt.plot(data['Time'], data['Score'])
plt.show()