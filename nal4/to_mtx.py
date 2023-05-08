from scipy.io import loadmat
import numpy as np

m = loadmat("mahalanobis.mat")
m_test = m["test"]
m_train = m["train"]

def to_arr(data):
    arr = [0]
    x, y = data.nonzero()
    for xs, ys in zip(x, y):
        arr.append(f"{xs+1} {ys+1} {data[xs, ys]}\n")

    (rows, cols) = np.shape(data)
    arr[0] = f"{rows} {cols} {len(arr)-1}\n"
    return arr

with open("test.mtx", "w") as f:
    f.writelines(to_arr(m_test))

with open("train.mtx", "w") as f:
    f.writelines(to_arr(m_train))
