import pickle
import numpy as np


def _cbc_min_max(path='data.pkl'):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    arr = []
    for d in data:
        arr.append(d['cbc'])
    arr = np.array(arr)
    return arr.min(), arr.max()


def _all_material_type():
    with open('_data.pkl', 'rb') as f:
        data = pickle.load(f)

    all_type = set()
    for d in data:
        materials = list(d['materials'].reshape(-1))
        for m in materials:
            all_type.add(m)
    print(all_type)


if __name__ == '__main__':
    _all_material_type()