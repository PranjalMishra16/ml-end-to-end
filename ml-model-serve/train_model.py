
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np


x = np.array ([[1],[2],[3],[4],[5],[6]])
y = np.array([8, 10,12,14,16,18])

model = LinearRegression()
model.fit(x,y)

with open ("model.pkl","wb") as f:
    pickle.dump(model, f)
    print('.pkl successfully created!')