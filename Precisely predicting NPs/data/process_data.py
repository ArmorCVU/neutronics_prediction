import numpy as np
import pickle
from tqdm.auto import tqdm


def read_csv_to_list(file):
    def line_to_float_list(line: str):
        """把一行的字符串转换成数组
        """
        nums = line.split(',')
        return [float(s) for s in nums]

    with open(file, 'r') as f:
        lines = []
        for line in f:
            float_list = line_to_float_list(line)
            assert len(float_list) == 225 + 1 + 52
            lines.append(float_list)
        return lines


def add_fq_quarter_to_data():
    with open('data.pkl', 'rb') as f:
        pkl_data = pickle.load(f)

    with open('data.csv', 'r') as f:
        for line, data in zip(f, pkl_data):
            line = line.split(',')
            quarter_fq = np.array(line[-52:], dtype=np.float32)
            data['quarter_fq'] = quarter_fq

    with open('data.pkl', 'wb') as f:
        pickle.dump(pkl_data, f)


def get_reactor_core(quarter):
    assert len(quarter) == 52
    valid_mask = [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # 0
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

    pos = 0
    fdh = np.zeros((15, 15), dtype=np.float32)
    # 填充右下角
    for i in range(7, 15):
        for j in range(7, 15):
            if valid_mask[i][j] == 1:
                fdh[i, j] = quarter[pos]
                pos += 1
    # 对称
    for i in range(0, 7):
        fdh[:, i] = fdh[:, 14 - i]

    for i in range(0, 7):
        fdh[i, :] = fdh[14 - i, :]

    return fdh


def process_data(file, output_file):
    data_list = read_csv_to_list(file)
    data = []

    for d in data_list:
        burn_type = d[:225]
        burn_type = np.array(burn_type).reshape(15, 15).astype(int)
        cbc = d[225]
        # quarter_fdh = np.array(d[226:]).astype(np.float32)
        quarter_fq = np.array(d[226:]).astype(np.float32)

        res = {
            'materials': burn_type,
            'cbc': cbc,
            # 'quarter_fdh': quarter_fdh,
            'quarter_fq': quarter_fq
        }
        data.append(res)

    with open(output_file, 'wb') as f:
        pickle.dump(data, f)


def process_material_mapping_csv():
    all_lines = []
    with open('material_mapping.csv', 'r') as f:
        for line in f:
            l = []
            for item in line.split(','):
                l.append(item.strip())
            all_lines.append(l)
    f.close()
    with open('mapping.csv', 'w') as f:
        for line in all_lines:
            for i, item in enumerate(line):
                f.write(item)
                if i != len(line) - 1:
                    f.write(',')
                else:
                    f.write('\n')
    f.close()


def statistic_material_tpyes():
    type_set = set()
    with open('mapping.csv', 'r') as f:
        for line in f:
            all_item = line.split(',')
            type_set.add(int(all_item[2]))
    print(type_set)


def process_data_with_mapping(data_file, mapping_file='mapping.csv'):
    # 加载 data 文件
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    # 加载 mapping
    mapping = []
    with open(mapping_file, 'r', encoding='utf-8') as f:
        for line in f:
            l = []
            for item in line.split(','):
                l.append(item)
            mapping.append(l)

    for d in tqdm(data):
        new_or_old = np.zeros((15, 15), int)
        material_type = np.zeros((15, 15), int)
        init_burn = np.zeros((15, 15), float)

        for i in range(15):
            for j in range(15):
                material_no = d['materials'][i][j]
                if material_no != 0:
                    # 每个组件的新旧信息 NEW => 0; OLD => 1
                    new_or_old[i][j] = 0 if mapping[material_no][1] == 'NEW' else 1
                    material_type[i][j] = int(mapping[material_no][2])
                    init_burn[i][j] = float(mapping[material_no][3])

        d['materials'] = material_type
        d['init_burn'] = init_burn
        d['new_or_old'] = new_or_old

    with open(data_file, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    process_data('fq.csv', 'data_fq.pkl')
    process_data_with_mapping(data_file='data_fq.pkl')