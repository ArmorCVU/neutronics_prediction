# 使用机器学习的回归模型拟合参数
import sklearn
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import torch
import time
from torch import Tensor
import torch.nn.functional as F
from tqdm.auto import tqdm

from dataset import rand_train_val_split, ReactorCoreDataset

# train/val split
train, val = rand_train_val_split()
train_size, val_size = len(train), len(val)
print(f'train_size={train_size}; val_size={val_size}')

# create dataset
train_data = ReactorCoreDataset(train, encoding='num')
valid_data = ReactorCoreDataset(val, encoding='num')

# train data
train_x = np.array([d['inp_tensor'] for d in train_data]).reshape(train_size, -1)
train_y_fq = np.array([d['fq'] for d in train_data]).reshape(train_size, -1)
train_y_fdh = np.array([d['fdh'] for d in train_data]).reshape(train_size, -1)
train_y_cbc = np.array([d['cbc'] for d in train_data]).reshape(train_size, -1)

# valid data
valid_x = np.array([d['inp_tensor'] for d in valid_data]).reshape(val_size, -1)
valid_y_fq = np.array([d['fq'] for d in valid_data]).reshape(val_size, -1)
valid_y_fdh = np.array([d['fdh'] for d in valid_data]).reshape(val_size, -1)
valid_y_cbc = np.array([d['cbc'] for d in valid_data]).reshape(val_size, -1)

# show shape
print(f'train_x shape: {train_x.shape}')
print(f'train_y_fq shape: {train_y_fq.shape}')
print(f'train_y_cbc shape: {train_y_cbc.shape}')


def mse_error(predict: np.ndarray, target: np.ndarray):
    predict = torch.from_numpy(predict)
    target = torch.from_numpy(target)
    return F.mse_loss(predict, target, reduction='mean')


def max_relative_error(predict: np.ndarray, target: np.ndarray):
    sum = 0.
    for p, t in zip(predict, target):
        max_pos = np.argmax(t)
        sum += np.abs(p[max_pos] - t[max_pos]) / t[max_pos]
    return sum / len(predict)


def fit_svm(train_x, train_y, valid_x, valid_y):
    from sklearn import svm
    regressor = svm.SVR(kernel='linear', C=1.0)
    regressor = MultiOutputRegressor(regressor)
    regressor.fit(train_x, train_y)
    predict_y = regressor.predict(valid_x)
    mse = mse_error(predict_y, valid_y)
    mre = max_relative_error(predict_y, valid_y)
    print(f'mse={mse}; mre={mre}')


def fit_random_forest(train_x, train_y, valid_x, valid_y):
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=1)
    regressor = MultiOutputRegressor(regressor)
    regressor.fit(train_x, train_y)
    predict_y = regressor.predict(valid_x)
    mse = mse_error(predict_y, valid_y)
    mre = max_relative_error(predict_y, valid_y)
    print(f'mse={mse}; mre={mre}')


def fit_linear_regression(train_x, train_y, valid_x, valid_y):
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor = MultiOutputRegressor(regressor)
    regressor.fit(train_x, train_y)
    predict_y = regressor.predict(valid_x)
    mse = mse_error(predict_y, valid_y)
    mre = max_relative_error(predict_y, valid_y)
    print(f'mse={mse}; mre={mre}')


def fit_rbf_regression(train_x, train_y, valid_x, valid_y):
    from sklearn import svm
    regressor = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    regressor = MultiOutputRegressor(regressor)
    regressor.fit(train_x, train_y)
    predict_y = regressor.predict(valid_x)
    mse = mse_error(predict_y, valid_y)
    mre = max_relative_error(predict_y, valid_y)
    print(f'mse={mse}; mre={mre}')


if __name__ == '__main__':
    # print(f'=======================================================================')
    # time_start = time.time()
    # fit_random_forest(train_x, train_y_fq, valid_x, valid_y_fq)
    # print(f'Durations: {time.time() - time_start: .1f}s\n')
    #
    # print(f'=======================================================================')
    # time_start = time.time()
    # fit_random_forest(train_x, train_y_fdh, valid_x, valid_y_fdh)
    # print(f'Durations: {time.time() - time_start: .1f}s\n')

    print(f'=======================================================================')
    train_x = train_x
    train_y_cbc = train_y_cbc
    time_start = time.time()
    fit_random_forest(train_x, train_y_cbc, valid_x, valid_y_cbc)
    print(f'Durations: {time.time() - time_start: .1f}s\n')
