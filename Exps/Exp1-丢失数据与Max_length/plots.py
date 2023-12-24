import pandas as pd
import matplotlib.pyplot as plt
recall_list = [0.0009800078400627205, 0.002048580626280363, 0.007725180802103879, 0.3604696673189824, 0.8382113821138212, 1.0, 0.001097608780870247, 0.002048580626280363, 0.007725180802103879, 0.3675146771037182, 0.8382113821138212, 1.0, 0.0008624068992551941, 0.002048580626280363, 0.007725180802103879, 0.3475538160469667, 0.8382113821138212, 1.0, 0.0006272050176401411, 0.002048580626280363, 0.007725180802103879, 0.3659491193737769, 0.8382113821138212, 1.0, 0.0007448059584476676, 0.002048580626280363, 0.007725180802103879, 0.361252446183953, 0.8382113821138212, 1.0]
precision_list = [0.20161290322580644, 0.2692307692307692, 0.27011494252873564, 0.24262381454162277, 0.32090887868648355, 1.0, 0.2, 0.2692307692307692, 0.27011494252873564, 0.24058416602613375, 0.32090887868648355, 1.0, 0.2, 0.2692307692307692, 0.27011494252873564, 0.2423580786026201, 0.32090887868648355, 1.0, 0.2077922077922078, 0.2692307692307692, 0.27011494252873564, 0.24241638579206637, 0.32090887868648355, 1.0, 0.2, 0.2692307692307692, 0.27011494252873564, 0.24270312910859848, 0.32090887868648355, 1.0]
f1_list = [0.001950534446438324, 0.004066221318617485, 0.015020773410035156, 0.2900330656589514, 0.4641269483990772, 1.0, 0.0021832358674463937, 0.004066221318617485, 0.015020773410035156, 0.29080210591514405, 0.4641269483990772, 1.0, 0.0017174082747853242, 0.004066221318617485, 0.015020773410035156, 0.2855764592378196, 0.4641269483990772, 1.0, 0.0012506350881306915, 0.004066221318617485, 0.015020773410035156, 0.2916406737367436, 0.4641269483990772, 1.0, 0.0014840851396211677, 0.004066221318617485, 0.015020773410035156, 0.29034287511796164, 0.4641269483990772, 1.0]
plt.style.use('seaborn-v0_8')
data = {
    "datasets": ['movies_1', 'hospital', 'beers', 'rayyan', 'flights', 'toy'],
    'F1-300': f1_list[0:6],
    'F1-400': f1_list[6:12],
    'F1-500':  f1_list[12:18],
    'F1-600': f1_list[18:24],
    'F1-700': f1_list[24:30]
}

df = pd.DataFrame(data)
print(df)
ax = df.plot(x='datasets', y=['F1-300','F1-400','F1-500','F1-600','F1-700'], kind='bar', figsize=(10, 5))
ax.set_title('F1-Score in different max_length')
plt.show()






impute_300 = [8,2,16,362,1908,6]
impute_400 = [9,2,16,372,1908,6]
impute_500 = [7,2,16,350,1908,6]
impute_600 = [6,2,16,363,1908,6]
impute_700 = [5,2,16,369,1908,6]
origin_data_len = [7390,1000,2410,1000,2376,6]
max_len = [300,400,500,600,700]
impute_movie = [8,9,7,6,5]
impute_hospital = [2,2,2,2,2]
impute_beers = [16,16,16,16,16]
impute_rayyan = [362,372,350,363,369]
impute_flights = [1908,1908,1908,1908,1908]
impute_toy = [6,6,6,6,6]
len_data = {
    'len': max_len,
    'movies': impute_movie,
    'hospital': impute_hospital,
    'beers': impute_beers,
    'rayyan': impute_rayyan,
    'flights': impute_flights,
    'toy': impute_toy
}
df_len = pd.DataFrame(len_data)
print(df_len)
ax = df_len.plot(x='len', y=['movies','hospital','beers','rayyan','flights','toy'], kind='line', figsize=(10, 5))
ax.set_title('Imputed data length in different max_length')
plt.show()



# 创建 3 行 2 列的子图
fig, axes = plt.subplots(3, 2, figsize=(10, 15))

# 为每个子图绘制线图
df_len.plot(ax=axes[0, 0], x='max_length', y='rayyan', kind='line', title='rayyan Imputed data length in different max_length')
df_len.plot(ax=axes[0, 1], x='max_length', y='movies', kind='line', title='movies Imputed data length in different max_length')
df_len.plot(ax=axes[1, 0], x='max_length', y='hospital', kind='line', title='hospital Imputed data length in different max_length')
df_len.plot(ax=axes[1, 1], x='max_length', y='beers', kind='line', title='beers Imputed data length in different max_length')
df_len.plot(ax=axes[2, 0], x='max_length', y='flights', kind='line', title='flights Imputed data length in different max_length')
df_len.plot(ax=axes[2, 1], x='max_length', y='toy', kind='line', title='toy Imputed data length in different max_length')

# 调整子图间距
plt.tight_layout()

# 保存图表到文件
file_path_pandas = 'imputed_data_length_plots_pandas.png'
plt.savefig(file_path_pandas)

# 显示图表
plt.show()