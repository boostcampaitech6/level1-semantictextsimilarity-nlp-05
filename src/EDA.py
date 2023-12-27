import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

import math

from scipy import stats
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu

# csv 파일 읽어와서 source 유형을 구분한 df 만듦.
def make_df(data_type):
   df = pd.read_csv(f'../must_not_upload/data/{data_type}.csv')
   
   source = df['source'].str.split('-')
   df['source_type'] = source.str.get(0)
   df['source_rtt'] = source.str.get(1)
   
   return df

# label 분포 box plot
def draw_box_plot(df, column):
    label_list = [df[column]]
    
    plt.boxplot(label_list)
    plt.title("Boxplot for target label")
    plt.show()

# label 분포 histogram1
def draw_hist(df):
   df['label'].hist()
   df['label'].hist(by=df['source_type']) # source_type에 따라 각각 histogram 그리기
   df['label'].hist(by=df['source_rtt']) # rtt/sampled 여부에 따라 각각 histogram 그리기
   plt.show()

# label 분포 histogram2
def draw_hist_plt(df):
    plt.hist(df['label'])

    plt.title('Histogram of Label')
    plt.xlabel('Label')
    plt.ylabel('Frequency')

    plt.show()

# label의 빈도 세기
def count_label(df):
    label_counts = df.groupby('label').agg({'label':['count']})
    label_counts['percent'] = round(label_counts[('label', 'count')]/sum(label_counts[('label', 'count')])*100, 1)
    
    print(label_counts)

# sentence1, 2 토크나이징 & 길이 차이 계산
def tokenizing(df):
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-small')

    id = list(df['id'])
    source = list(df['source'])
    sentence1_input = list(df['sentence_1'])
    sentence2_input = list(df['sentence_2'])
    source_rtt = list(df['source_rtt'])
    label = list(df['label'])
    binary_label = list(df['binary-label'])

    sentence1_sentence2_len = []
    sentence1_sentence2_unk = []

    for i, item in df.iterrows():
        sentence1 = tokenizer(item['sentence_1'])['input_ids']
        sentence2 = tokenizer(item['sentence_2'])['input_ids']

        sentence1_sentence2_len_diff = len(sentence1)-len(sentence2)
        sentence1_sentence2_unk_diff = sentence1.count(tokenizer.unk_token_id)-sentence2.count(tokenizer.unk_token_id)

        sentence1_sentence2_len.append(sentence1_sentence2_len_diff)
        sentence1_sentence2_unk.append(sentence1_sentence2_unk_diff)

    tokenized_df = pd.DataFrame([id, source, sentence1_input, sentence2_input, source_rtt, label, binary_label, 
                                 sentence1_sentence2_len, sentence1_sentence2_unk]).transpose()
    tokenized_df.columns = ['id', 'source', 'sentence_1', 'sentence_2', 'source_rtt', 'label', 'binary-label', '1_len - 2_len', '1_unk - 2_unk']

    return tokenized_df

# 데이터 불러오기
train_df = make_df('train')

## label 분포 시각화
# 박스플롯
draw_box_plot(train_df, 'label')

# 히스토그램
draw_hist(train_df)
draw_hist_plt(train_df)

## 문장 길이 차이 시각화
tokenized_train_df = tokenizing(train_df)

draw_box_plot(tokenized_train_df, '1_len - 2_len')

tokenized_train_df['1_len - 2_len'] = tokenized_train_df['1_len - 2_len'].astype(float)

# 확률밀도함수 그래프
sns.kdeplot(data=tokenized_train_df['1_len - 2_len'], color = 'red', fill=True)
plt.title("Density plot")
plt.xlabel("Sentence Length Difference")
plt.show()

# QQ plot
(stats.probplot(tokenized_train_df['1_len - 2_len'], dist="norm", plot=plt))
plt.title("Normal Q-Q plot")
plt.ylabel("Sample Quantiles")
plt.xlabel("Theoretical Quantiles")
plt.show()

## 정규성 검정: 스미르노프(Kolmogorov-Smirnov) 검정
ks_test_1_len_2_len = stats.kstest(tokenized_train_df['1_len - 2_len'], 'norm', args=(0, 1))
print('ks_test_1_len_2_len: ', ks_test_1_len_2_len)