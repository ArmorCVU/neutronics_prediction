import sys

import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy.random as npr
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split


def rand_train_val_split(path=r'data/data.pkl', train_size=0.8):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    train, val = train_test_split(data, train_size=train_size)
    return train, val


class ReactorCoreDataset(Dataset):
    MATERIAL_VALUE = ['44508', '44512', '44516']
    TYPE_NUM_MAP = {44508: 0, 44512: 1, 44516: 2}

    CBC_MIN = 1737.3
    CBC_MAX = 2008.5

    INIT_BURN_MIN = 0.
    INIT_BURN_MAX = 38366.

    FDH_MIN = 0.2655
    FDH_MAX = 1.9331

    FQ_MIN = 0.3388
    FQ_MAX = 2.6175

    # 这是一个 15 * 15 的矩阵, 其中如果某个位置是 1, 那么表示这个位置是一个放置燃料的有效位置.
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

    def __init__(self, data, cal_min_max=False):
        self.data = data
        if cal_min_max:
            self._get_min_max()
        self._process_data()

    def _get_min_max(self):
        # 计算 cbc 的最大值和最小值
        if isinstance(self.CBC_MIN, type(...)) or self.CBC_MIN is None:
            arr = []
            for d in self.data:
                arr.append(d['cbc'])
            arr = np.array(arr)
            self.CBC_MIN = arr.min()
            self.CBC_MAX = arr.max()
        print(f'cbc min: {self.CBC_MIN}, cbc max: {self.CBC_MAX}')

        # 计算 init_burn 的最大值和最小值
        if isinstance(self.INIT_BURN_MIN, type(...)) or self.INIT_BURN_MIN is None:
            all_values = []
            for d in self.data:
                mat = d['init_burn']
                for i in range(15):
                    for j in range(15):
                        if ReactorCoreDataset.VALID_POSITION_MASK[i][j] == 1:
                            all_values.append(mat[i][j])
            all_values = np.array(all_values)
            self.INIT_BURN_MIN = all_values.min()
            self.INIT_BURN_MAX = all_values.max()

        # 计算 fq 的最大值和最小值
        if isinstance(self.FQ_MIN, type(...)) or self.FQ_MIN is None:
            fq_min = 99999999
            fq_max = -fq_min

            for d in self.data:
                fq_min = min(fq_min, d['quarter_fq'].min())
                fq_max = max(fq_max, d['quarter_fq'].max())

            self.FQ_MIN = fq_min
            self.FQ_MAX = fq_max


        # 计算 fdh 的最大值和最小值
        if isinstance(self.FDH_MIN, type(...)) or self.FDH_MIN is None:
            fdh_min = 99999999
            fdh_max = -fdh_min

            for d in self.data:
                fdh_min = min(fdh_min, d['quarter_fdh'].min())
                fdh_max = max(fdh_max, d['quarter_fdh'].max())

            self.FDH_MIN = fdh_min
            self.FDH_MAX = fdh_max

    def _convert_quarter_fq(self, quarter):
        """原始数据只有 fq 的右下 1/4. 并且这 1/4 的数据是一维的.
        这个函数将这些数据处理成 15 * 15 的矩阵.
        """
        assert len(quarter) == 52
        pos = 0
        fq = np.zeros((15, 15), dtype=np.float32)
        # 填充右下角
        for i in range(7, 15):
            for j in range(7, 15):
                if self.VALID_POSITION_MASK[i][j] == 1:
                    fq[i, j] = quarter[pos]
                    pos += 1
        # 对称
        for i in range(0, 7):
            fq[:, i] = fq[:, 14 - i]

        for i in range(0, 7):
            fq[i, :] = fq[14 - i, :]

        return fq

    def _convert_quarter_fdh(self, quarter):
        """原始数据只有 fdh 的右下 1/4. 并且这 1/4 的数据是一维的.
        这个函数将这些数据处理成 15 * 15 的矩阵.
        """
        assert len(quarter) == 52
        pos = 0
        fdh = np.zeros((15, 15), dtype=np.float32)
        # 填充右下角
        for i in range(7, 15):
            for j in range(7, 15):
                if self.VALID_POSITION_MASK[i][j] == 1:
                    fdh[i, j] = quarter[pos]
                    pos += 1
        # 对称
        for i in range(0, 7):
            fdh[:, i] = fdh[:, 14 - i]

        for i in range(0, 7):
            fdh[i, :] = fdh[14 - i, :]

        return fdh


    def _min_max_process_init_burn(self, init_burn):
        """将 init_burn 进行 min-max 归一化
        """
        return (init_burn - self.INIT_BURN_MIN) / (self.INIT_BURN_MAX - self.INIT_BURN_MIN)

    def _min_max_process_cbc(self, cbc):
        """将 cbc 进行 min-max 归一化
        """
        return (cbc - self.CBC_MIN) / (self.CBC_MAX - self.CBC_MIN)

    def _min_max_process_fq(self, fq):
        """将 fq 进行 min-max 归一化
        """
        return (fq - self.FQ_MIN) / (self.FQ_MAX - self.FQ_MIN) * np.array(self.VALID_POSITION_MASK)

    def _min_max_process_fdh(self, fdh):
        """将 fdh 进行 min-max 归一化
        """
        return (fdh - self.FDH_MIN) / (self.FDH_MAX - self.FDH_MIN) * np.array(self.VALID_POSITION_MASK)


    def _process_data(self):
        _cls = self.__class__.__name__
        for d in tqdm(self.data, desc=f'{_cls} processing data'):
            # 对初始燃耗进行 min-max 归一化
            d['init_burn'] = self.VALID_POSITION_MASK * d['init_burn']
            d['init_burn'] = self._min_max_process_init_burn(d['init_burn'])

            # 对 CBC 进行 min-max 归一化
            d['cbc'] = np.float32(self._min_max_process_cbc(d['cbc']))

            # 处理 fq
            fq = self._convert_quarter_fq(d['quarter_fq'])
            fq = self._min_max_process_fq(fq).astype(np.float32)
            d['fq'] = fq

            # 处理一下 fdh
            fdh = self._convert_quarter_fdh(d['quarter_fdh'])
            fdh = self._min_max_process_fdh(fdh).astype(np.float32)
            d['fdh'] = fdh

            # 以后将会转化成 5 层 input_Tensor,包含了材料层[0, 1, 2], 燃耗层[3], 新旧层[4]
            inp_tensor = np.zeros(shape=(5, 15, 15))

            # 处理 materials 并将其放入 inp_tensor
            materials = d['materials']
            for i in range(len(materials)):
                for j in range(len(materials)):
                    material_type = materials[i][j]
                    if material_type != 0:
                        type_num = self.TYPE_NUM_MAP[material_type]
                        inp_tensor[type_num][i][j] = 1

            # 处理初始燃耗并将其放入 inp_tensor
            inp_tensor[3] = d['init_burn']

            # 将新旧组件信息放入 inp_tensor
            inp_tensor[4] = d['new_or_old']

            # 保存当前数据的 inp_tensor
            d['inp_tensor'] = inp_tensor.astype(np.float32)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def __print_statistic():
    """打印一些统计信息.
    """
    def _cbc_min_max(path=r'data/data.pkl'):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        arr = []
        for d in data:
            arr.append(d['cbc'])
        arr = np.array(arr)
        return arr.min(), arr.max()

    def _init_burn_min_max(path=r'data/data.pkl'):
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

    def _fq_min_max():
        with open(r'data/data.pkl', 'rb') as f:
            data = pickle.load(f)
        fq_min = 99999999
        fq_max = -fq_min

        for d in data:
            fq_min = min(fq_min, d['quarter_fq'].min())
            fq_max = max(fq_max, d['quarter_fq'].max())

        return fq_min, fq_max

    def _fdh_min_max(path=r'data/data.pkl'):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        fdh_min = 99999999
        fdh_max = -fdh_min

        for d in data:
            fdh_min = min(fdh_min, d['quarter_fdh'].min())
            fdh_max = max(fdh_max, d['quarter_fdh'].max())

        return fdh_min, fdh_max

    # 打印 CBC 数据的最小值和最大值
    minv, maxv = _cbc_min_max()
    print(f'min cbc: {minv}, max cbc: {maxv}')
    # 打印初始燃耗的最小值和最大值
    minv, maxv = _init_burn_min_max()
    print(f'min init burn: {minv}, max init burn: {maxv}')
    # 打印  fq 的最小值和最大值
    minv, maxv = _fq_min_max()
    print(f'min fq: {minv}. max fq: {maxv}')
    # 打印 fdh 的最小值和最大值
    minv, maxv = _fdh_min_max()
    print(f'min fdh: {minv}, max fdh: {maxv}')



if __name__ == '__main__':
    def print_matrix(matrix):
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                print(f'{matrix[i][j]: 6f} ', end='')
                if j == len(matrix[0]) - 1:
                    print()

    train, val = rand_train_val_split()
    d = ReactorCoreDataset(val)
    fdh = d[0]['fdh']
    print_matrix(fdh)
    print(d[0]['fdh'].shape)

    __print_statistic()

    # train, val = rand_train_val_split()
    # d = ReactorCoreDataset(val)
    # print(d[0]['cbc'])



