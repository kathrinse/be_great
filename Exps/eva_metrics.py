import pandas as pd  
import numpy as np
import logging
import os
def calculate_precision_recall(data_df, label_df, result_df):  
    # 确保DlocaFrame的索引一致  
    if data_df.index.nlevels != 1 or label_df.index.nlevels != 1 or result_df.index.nlevels != 1:  
        raise ValueError("DlocaFrames must have a single level index.")  
    tp = 0  # True Positives  
    fp = 0  # False Positives  
    fn = 0  # False Neglocives  
    count,edit = 0,0
    missing_keys = set()
    for index  in data_df.index : 
        for col in  data_df.columns :
            logging.info(f"index={index},col={col}")
            label = label_df.loc[index, col]
            data = data_df.loc[index, col]
            try:
                result = result_df.loc[index,col]
            except:
                # logging.warning(f"key {index} not found.")
                missing_keys.add(index)
                continue
            count += 1
            logging.debug(f"data={data},label={label},result={result}")
            # 比较修改后的数据点和标签
            if data != result:
                edit += 1
                # 如果修改后的数据点与标签匹配，则增加TP计数
                if result == label:
                    tp += 1
                # 如果修改后的数据点与标签不匹配，则增加FP计数
                if result != label:
                    fp += 1
                # 如果原始数据点与标签不匹配，但修改后匹配，则增加FN计数
            if data != label and result == data: # 需要改但没改的
                    fn += 1

  
    # 计算Precision和Recall  
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0  
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  
    logging.warning(f"补全后的结果丢失数据{len(missing_keys)}条，dirty中有{len(data_df)}条数据，补全后为{len(result_df)}")
    logging.debug(f"丢失的数据分别是{missing_keys}")
    logging.info(f"tp={tp},fp={fp},fn={fn},count={count},edit={edit}")
    return precision, recall
def process_beers(data,label,result):
    """专用于为beers数据集计算precision和recall
        做的处理有：
        - 取id列作为索引
        - 将一些补全后成为浮点的数据转换为整数
        
    Args:
        data_df (pd.Dataframe): 原始脏数据的df
        label_df (pd.Dataframe): 干净数据的df
        result_df (pd.Dataframe): 补全完的df
    
    Return:
        - precision
        - recall
    """
    result.drop(['Unnamed: 0'], axis=1, inplace=True)
    result['id'] = result['id'].astype('int')
    result['brewery_id'] = result['brewery_id'].astype('int')
    result['index'] = result['index'].astype('int')
    result['ibu'] = result['ibu'].astype('int')
    result.set_index('index',inplace=True)

    data.set_index('index',inplace=True)

    label.set_index('index',inplace=True)
    precision,recall = calculate_precision_recall(data, label, result)
    return precision,recall
def process_ave(data,label,result):
    """适用于一般数据集计算precision和recall
        做的处理有：
        - 去除补全结果保存时多余的一列
        
    Args:
        data_df (pd.Dataframe): 原始脏数据的df
        label_df (pd.Dataframe): 干净数据的df
        result_df (pd.Dataframe): 补全完的df
    
    Return:
        - precision
        - recall
    """
    result.drop(['Unnamed: 0'], axis=1, inplace=True)
    precision,recall = calculate_precision_recall(data, label, result)
    return precision,recall


if __name__ == "__main__":
    # result = pd.read_csv('exp3/exp2.5/beer_imputed_epoches-1.csv')
    # data = pd.read_csv('raha/datasets/beers/dirty.csv')
    # label = pd.read_csv('raha/datasets/beers/clean.csv')
    # ans = process_beers(data, label, result)
    # print(ans)
    data_dir = r"E:\project\raha\datasets"
    file_path = r"D:\Desktop\GReat\Exp2-输入数据不做任何处理\补全结果"
    file_dir = os.listdir(file_path)
    dataset_list = [dataset.split('_')[0] for dataset in file_dir]
    results = []
    for dataset in dataset_list:
        if dataset == 'movies':
            dataset = 'movies_1'
        data = pd.read_csv(os.path.join(data_dir,dataset,'dirty.csv'))
        label = pd.read_csv(os.path.join(data_dir,dataset,'clean.csv'))
        result = pd.read_csv(os.path.join(file_path,dataset+'_imputed.csv'))
        if dataset == 'beers':
            precision,recall = process_beers(data, label, result)
        else:
            try:
                precision,recall = process_ave(data, label, result)
            except:
                precision,recall = 0,0
                print(dataset)
        results.append({'dataset':dataset,'precision':precision,'recall':recall})
        print(f"{dataset}的precision={precision},recall={recall}")
    # print(dataset_list)
    df = pd.DataFrame.from_dict(results)
    print(df.to_markdown())

