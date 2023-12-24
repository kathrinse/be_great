import pandas as pd
import numpy as np
import os

from scipy import datasets
from be_great import GReaT

datasets_path = "/root/raha/datasets/"
# datasets =  os.listdir(datasets_path)
datasets = ['movies_1', 'hospital', 'beers', 'rayyan', 'flights', 'toy']
print("datasets:",datasets)
imputed_data_len_list = []
dirty_len_list = []
tuple_index_len_list = []
right_list = []
all_list = []
count_list = []
recall_list = []
precision_list = []
f1_list = []
for max_length in range (300,1100,100):
    for dataset in datasets:
        if dataset == 'tax':
            continue
        print("⭐⭐⭐正在处理数据集：",dataset)
        # 读取两个CSV文件
        clean = pd.read_csv(datasets_path + dataset+'/clean.csv')
        dirty = pd.read_csv(datasets_path + dataset+'/dirty.csv')
        # 手动把dirty中的错误值替换为np.nan
        row_index = []
        col_index = []
        tuple_index = []
        clean_dict = {}
        dirt_dict = {}
        # print(clean.iloc[1,6])
        # print(dirty.iloc[1,6])
        for i in range(len(clean)):
            for j in range(len(clean.columns)):
                if clean.iloc[i, j] != dirty.iloc[i, j]:
                    row_index.append(i)
                    col_index.append(j)
                    tuple_index.append((i, j))
                    clean_dict[(i, j)] = clean.iloc[i, j]
                    dirt_dict[(i, j)] = dirty.iloc[i, j]
                    dirty.iloc[i, j] = np.nan
        # print(tuple_index)
        # print(dirty.iloc[1,6])
        # print(clean_dict)

        model = GReaT(llm='distilgpt2', batch_size=32, epochs=25)
        model.fit(dirty)
        # synthetic_data = model.sample(n_samples=100)
        # print(synthetic_data.head())
        try:
            imputed_data = model.impute(dirty, max_length=max_length,temperature=0.1,max_retries=15,k=150) # 默认的max_length是200
        except Exception as e:
            print(e)
            print(f"⚠⚠⚠在max_length={max_length}时，处理数据集{dataset}时，发生如上错误，跳过")
            continue
        imputed_data.to_csv(dataset + '_imputed.csv')
        print("补全前的数据尺寸（行，列）：",dirty.shape)
        print("补全后的数据尺寸（行，列）：",imputed_data.shape)

        right = 0 # 统计补全后的数据与clean中相同的单元格个数，即正确修改的个数
        all = len(tuple_index) # 统计dirty中的错误值的个数，即需要修改的个数
        try: # 补全后的数据可能会比dirty少若干行，且少的行数不固定
            for tup in tuple_index:
                if imputed_data.iloc[tup[0], tup[1]] == clean.iloc[tup[0], tup[1]] :
                    print(imputed_data.iloc[tup[0], tup[1]],clean.iloc[tup[0], tup[1]])
                right += 1
        except:
            pass
        recall = right / all
        
        # 统计imputed_data与dirty不同单元格的个数
        count = 0
        for i in range(min(len(dirty), len(imputed_data))):
            for j in range( min( len(dirty.columns), len(imputed_data.columns))):
                if dirty.iloc[i, j] != imputed_data.iloc[i, j]:
                    count += 1
        precision = right / count
        f1 = 2 * recall * precision / (recall + precision)
        print(f"imputed_data_len:{len(imputed_data)},dirty_len:{len(dirty)},tuple_index_len:{len(tuple_index)},right:{right},all:{all},count:{count}")
        print(f"recall:{recall},precision:{precision},f1:{f1}")
        imputed_data_len_list.append(len(imputed_data))
        dirty_len_list.append(len(dirty))
        tuple_index_len_list.append(len(tuple_index))
        right_list.append(right)
        all_list.append(all)
        count_list.append(count)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)
    print(f"🔥🔥🔥 max_length = {max_length}")
    print(datasets)
    print(recall_list)
    print(precision_list)
    print(f1_list)