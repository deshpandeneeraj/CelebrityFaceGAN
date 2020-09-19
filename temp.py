from sklearn.utils import shuffle
import numpy as np
X = np.array([[1,2,3],[4,5,6]])
y = np.array([8,9])


X, y = shuffle(X, y, random_state=0)
print(f'X = {X}\ny = {y}')