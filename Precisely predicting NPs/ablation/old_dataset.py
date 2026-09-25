import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy.random as npr
from sklearn.model_selection import train_test_split


def rand_train_val_split(path='old_data/data.pkl', train_size=0.8):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    train, val = train_test_split(data, train_size=train_size)
    return train, val


def _cbc_min_max(path='old_data/data.pkl'):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    arr = []
    for d in data:
        arr.append(d['cbc'])
    arr = np.array(arr)
    return arr.min(), arr.max()


def _init_burn_min_max(path='old_data/data.pkl'):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    all_values = []
    for d in data:
        mat = d['init_burn']
        for i in range(15):
            for j in range(15):
                if ReactorCoreDataset.VALID_POSITION_MASK[i][j] == 1:
                    all_values.append(mat[i][j])
    all_values = np.array(all_values)
    return all_values.min(), all_values.max()


class ReactorCoreDataset(Dataset):
    TYPE_VALUE_NUM = [18000, 24004, 24008, 31000, 31008, 31012, 31016, 39000, 44512, 44516, 44520, 49512]

    TYPE_NUM_MAP = {
        18000: 0, 24004: 1, 24008: 2, 31000: 3, 31008: 4, 31012: 5, 31016: 6, 39000: 7, 44512: 8, 44516: 9,
        44520: 10, 49512: 11
    }

    CBC_MIN = 601.6
    CBC_MAX = 3013.6

    INIT_BURN_MIN = 0.
    INIT_BURN_MAX = 20128.

    VALID_POSITION_MASK = [
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # 0
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # 1
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # 2
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 3
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 4
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 5
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 6
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 7
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 8
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 9
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 10
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 11
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # 12
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # 13
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]  # 14

    INVALID_POSITION_MASK = -np.array(VALID_POSITION_MASK, int) + 1

    def _cal_mean_std(self, key: str):
        all_values = []
        for d in self.data:
            mat = d[key]
            for i in range(15):
                for j in range(15):
                    if self.VALID_POSITION_MASK[i][j] == 1:
                        all_values.append(mat[i][j])
        all_values = np.array(all_values)
        return all_values.mean(), all_values.std()

    # def _normalize_matrix(self, mat, mean, std):
    #     ret = (mat - mean) / std
    #     ret = ret * self.VALID_POSITION_MASK
    #     return ret

    # def _denormalize_matrix(self, mat, mean, std, mask):
    #     ret = mat * std + mean
    #     if mask:
    #         ret = ret * self.VALID_POSITION_MASK
    #     return ret

    def __init__(self, data, normalize=True):
        self.data = data
        # self.normalize = normalize
        #
        self.fq_mean, self.fq_std = self._cal_mean_std('fq')
        self.fdh_mean, self.fdh_std = self._cal_mean_std('fdh')
        self.cbc_mean = np.array([d['cbc'] for d in self.data]).mean()
        self.cbc_std = np.array([d['cbc'] for d in self.data]).std()
        self.init_burn_mean, self.init_burn_std = self._cal_mean_std('init_burn')
        #
        self._process_data()

    # def normalize_fq(self, fq):
    #     return self._normalize_matrix(fq, self.fq_mean, self.fq_std)
    #
    # def denormalize_fq(self, fq, mask=False):
    #     return self._denormalize_matrix(fq, self.fq_mean, self.fq_std, mask)
    #
    # def normalize_fdh(self, fdh):
    #     return self._normalize_matrix(fdh, self.fdh_mean, self.fdh_std)
    #
    # def denormalize_fdh(self, fdh, mask=False):
    #     return self._denormalize_matrix(fdh, self.fdh_mean, self.fdh_std, mask)
    #
    # def normalize_cbc(self, cbc):
    #     return (cbc - self.cbc_mean) / self.cbc_std
    #
    # def denormalize_cbc(self, cbc):
    #     return cbc * self.cbc_std + self.cbc_mean

    def _min_max_process_init_burn(self, init_burn):
        return (init_burn - self.INIT_BURN_MIN) / (self.INIT_BURN_MAX - self.INIT_BURN_MIN)

    def _recover_min_max_process_init_burn(self, init_burn):
        return init_burn * (self.INIT_BURN_MAX - self.INIT_BURN_MIN) + self.INIT_BURN_MIN

    def _min_max_process_cbc(self, cbc):
        return (cbc - self.CBC_MIN) / (self.CBC_MAX - self.CBC_MIN)

    def _recover_min_max_process_cbc(self, cbc):
        return cbc * (self.CBC_MAX - self.CBC_MIN) + self.CBC_MIN

    # def normalize_init_burn(self, init_burn):
    #     return self._normalize_matrix(init_burn, self.fdh_mean, self.fdh_std)
    #     # return init_burn / self.init_burn_max

    # def denormalize_init_burn(self, init_burn, mask=False):
    #     return self._denormalize_matrix(init_burn, self.fdh_mean, self.fdh_std, mask)
    #     # return init_burn * self.init_burn_max

    def _process_data(self):
        for d in self.data:
            # 对初始燃耗进行 min-max 归一化
            d['init_burn'] = self.VALID_POSITION_MASK * d['init_burn']
            d['init_burn'] = self._min_max_process_init_burn(d['init_burn'])
            # 对 CBC 进行 min-max 归一化
            d['cbc'] = self._min_max_process_cbc(d['cbc'])

            # if self.normalize:
            #     # 先对所有数据进行规范化
            #     d['fq'] = self.normalize_fq(d['fq'])
            #     d['fdh'] = self.normalize_fdh(d['fdh'])
            #     d['cbc'] = self.normalize_cbc(d['cbc'])
            #     d['init_burn'] = self.normalize_init_burn(d['init_burn'])

            d['fq'] = d['fq'].astype(np.float32)
            d['fq_max'] = d['fq'].max()
            d['fdh'] = d['fdh'].astype(np.float32)
            d['fdh_max'] = d['fdh'].max()
            d['cbc'] = d['cbc'].astype(np.float32)
            d['init_burn'] = d['init_burn'].astype(np.float32)

            # 以后将会转化成 13 层 Tensor
            # 其中 0-11 层表示每个堆芯材料的位置. 第 12 层表示初始燃耗
            inp_tensor = np.zeros(shape=(13, 15, 15), dtype=np.float32)

            # 处理 materials 并将其放入 inp_tensor
            materials = d['materials']
            for i in range(len(materials)):
                for j in range(len(materials)):
                    material_type = materials[i][j]
                    if material_type != 0:
                        type_num = self.TYPE_NUM_MAP[material_type]
                        inp_tensor[type_num][i][j] = 1

            # 处理初始燃耗并将其放入 inp_tensor
            init_burn = d['init_burn']
            inp_tensor[12] = init_burn

            # 保存当前数据的 inp_tensor
            d['inp_tensor'] = inp_tensor

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    # 打印 CBC 数据的最小值和最大值
    minv, maxv = _cbc_min_max()
    print(f'min cbc: {minv}, max cbc: {maxv}')

    minv, maxv = _init_burn_min_max()
    print(f'min init burn: {minv}, max init burn: {maxv}')

    # train, val = rand_train_val_split()
    # data = ReactorCoreDataset(train, normalize=False)
    # for d in data:
    #     print(d['cbc'])
