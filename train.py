from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import pandas as pd
import numpy as np
import pickle

iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data']], columns=iris['feature_names'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
X = df.iloc[:,:4].values
y=df['species']

model = Perceptron()
model.fit(X, y)

with open("model.pkl", "wb") as fh:
    pickle.dump(model, fh)
