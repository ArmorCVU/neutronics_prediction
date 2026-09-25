from dataset import *


# dataset
train, val = rand_train_val_split()
# train_data = ReactorCoreDataset(train)
valid_data = ReactorCoreDataset(val)
# train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
# valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE)

cbcs =[vd['cbc'] for vd in valid_data]
print(max(cbcs))
print(min(cbcs))
print(valid_data[0]['cbc'])


# import model, time, torch
# device = 'cuda:0'
# input = torch.from_numpy(valid_data[0]['inp_tensor']).unsqueeze(0).to(device)
# model = model.Model(5, include_top=False).to(device)
# model.eval()
# st = time.time()
# for i in range(10):
#     out = model(input)
# print(f'avg inference time: {(time.time() - st) / 10}s')
