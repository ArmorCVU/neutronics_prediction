import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(9, 6))
plt.style.use('seaborn-whitegrid')
criterions = ['mse','bpr','listmle','wpair2']
lbs = ['Pointwise (MSE)','Pairwise (BPR)','Listwise (ListMLE)','Weighted (WHR2)']
dict_list=[]
for loss in criterions:
    res = get_res(loss,'cf10','201','rea')
    dict={
        'loss':loss,
        'acc_mean':res[0]*100,
        'acc_std': res[1]*100,
        'err_mean': (1-res[0]) * 100,
        'err_std': res[1] * 100,
        'tau_mean':res[2],
        'tau_std': res[3],
        'wtau_mean': res[4],
        'wtau_std': res[5],
        'N_10_mean': res[6],
        'N_10_std': res[7],
        'Rel_10_mean': res[8],
        'Rel_10_std': res[9],
    }
    dict_list.append(dict)


size = [20,30,40,50,60,70,80,90,100]
colors = ['#17becf','#bcbd22','#fa83d5','#ff7f0e','#d62728','#2ca02c','#1f77b4','#9467bd','#8c564b','#7f7f7f']

plt.plot(size, dict_list[0]['err_mean'], label=lbs[0], color=colors[0], markeredgecolor=colors[0], markerfacecolor='none', marker='o', markersize='5', linewidth=1.5)
plt.fill_between(size, dict_list[0]['err_mean'] - dict_list[0]['err_std'], dict_list[0]['err_mean'] + dict_list[0]['err_std'], color=colors[0], alpha=0.15)
plt.plot(size, dict_list[1]['err_mean'], label=lbs[1], color=colors[3], markeredgecolor=colors[3], markerfacecolor='none', marker='s', markersize='5', linewidth=1.5)
plt.fill_between(size, dict_list[1]['err_mean'] - dict_list[1]['err_std'], dict_list[1]['err_mean'] + dict_list[1]['err_std'], color=colors[3], alpha=0.15)
plt.plot(size, dict_list[2]['err_mean'], label=lbs[2], color=colors[5], markeredgecolor=colors[5], markerfacecolor='none', marker='^', markersize='5', linewidth=1.5)
plt.fill_between(size, dict_list[2]['err_mean'] - dict_list[2]['err_std'], dict_list[2]['err_mean'] + dict_list[2]['err_std'], color=colors[5], alpha=0.15)
plt.plot(size, dict_list[3]['err_mean'], label=lbs[3], color=colors[4], markeredgecolor=colors[4], markerfac