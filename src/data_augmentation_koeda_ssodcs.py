import pandas as pd
import numpy as np
from transformers import AutoTokenizer

import regex as re
from konlpy.tag import Okt

from koeda import EDA
from tqdm import tqdm

# csv 파일 DataFrame으로 불러오기
# 해당 csv 파일 데이터에 맞춤법 제거가 선행되어야 함.
def make_df(data):
    df = pd.read_csv(f'../must_not_upload/data/{data}.csv')

    source = df['source'].str.split('-')
    df['source_type'] = source.str.get(0)
    df['source_rtt'] = source.str.get(1)

    return df

# 불용어 제거
def custom_sentence(Sentence):      #전처리

    Sentence=re.sub('[.!\^]' , ' ',Sentence)
    Sentence=re.sub('[ㄱ-ㅎㅏ-ㅣ]|[^가-힣a-zA-Z0-9~\?\s:%]' , ' ',Sentence)  #특수문자들 몇개뺴고 처리

    Sentence=re.sub('(\?[^\?가-힣a-zA-Z0-9]*)' , '?',Sentence)    #? 뒤에있는애들 다 없애기
    Sentence=re.sub('([^\?가-힣a-zA-Z0-9]*\?)' , '?',Sentence)    #? 앞에있는애들 다 없애기

    Sentence=re.sub('([^가-힣a-zA-Z0-9])\\1{1,}' , '\\1',Sentence)   #반복되는 특수문자들, 띄어쓰기
    Sentence=re.sub('([가-힣a-zA-Z])\\1{4,}' , '\\1'*3,Sentence)    #반복되는 글자들 처리
    Sentence=re.sub('^[~?\s:%]*|[~?\s:%]*$', '',Sentence)    #문장 처음이 띄어쓰기면 없애줌.

    Sentence=Sentence

    if not Sentence:Sentence=' '

    return Sentence

def drop_none(df):
    df = df.dropna(subset=['sentence_1'], axis=0)
    df = df.dropna(subset=['sentence_2'], axis=0)
    return df

def remove_stopword(df, *col_list):
    df = drop_none(df)
    for col in col_list:
        df[col] = df[col].apply(custom_sentence)

    return df

# 이상치 제거
def remove_outliers(df, column):
    # 1사분위수(Q1)와 3사분위수(Q3) 계산
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # 이상치 기준 정의
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 이상치 제거
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    outlier_df = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return filtered_df, outlier_df

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

# 랜덤 샘플링
def sampling_func(data, sample_pct):
    np.random.seed(123)
    N = len(data)
    sample_n = int(len(data)*sample_pct)
    sample = data.take(np.random.permutation(N)[:sample_n])
    return sample

# copy sentence
def copy_sentence(df):
    sentences = list(set(list(df['sentence_1']) + list(df['sentence_2'])))
    id = [idx for idx in range(len(sentences))]
    source = ['copy'] * len(sentences)
    sentences_label = [5.0] * len(sentences)
    sentences_binary_label = [1.0] * len(sentences)
    source_rtt = df['source_rtt']
    
    products_list = [id, source, sentences, sentences, sentences_label, sentences_binary_label, source_rtt]
    
    sentences_df = pd.DataFrame(products_list).transpose()
    sentences_df.columns = ['id', 'source', 'sentence_1', 'sentence_2', 'label', 'binary-label', 'source_rtt']
    
    return sentences_df

# down sampling + copy sentence
def down_copy_sentence(df):

  # down
    df_0 = df[(df['label'] == 0)]
    filtered_df = df[(df['label'] == 0) & (df['source_rtt'] == 'sampled')]
    pct = (len(df_0) - 1200) / len(filtered_df)

    sampled_set = filtered_df.apply(sampling_func, sample_pct=pct)
    sampled_set_idx = sampled_set.index.to_list()
    df.drop(sampled_set_idx, axis=0, inplace=True)

    # copy
    sentences = list(set(list(sampled_set['sentence_1']) + list(sampled_set['sentence_2'])))
    id = [idx for idx in range(len(sentences))]
    source = ['down_copy'] * len(sentences)
    sentences_label = [5.0] * len(sentences)
    sentences_binary_label = [1.0] * len(sentences)
    source_rtt = sampled_set['source_rtt']

    products_list = [id, source, sentences, sentences, sentences_label, sentences_binary_label, source_rtt]

    sentences_df = pd.DataFrame(products_list).transpose()
    sentences_df.columns = ['id', 'source', 'sentence_1', 'sentence_2', 'label', 'binary-label', 'source_rtt']

    return sentences_df

# swap_sentence
def swap_sentence(df):
    sentences_1 = df['sentence_2']
    sentences_2 = df['sentence_1']
    id = [idx for idx in range(len(sentences_1))]
    source = ['copy'] * len(sentences_1)
    sentences_label = df['label']
    sentences_binary_label = df['binary-label']
    source_rtt = df['source_rtt']

    products_list = [id, source, sentences_1, sentences_2, sentences_label, sentences_binary_label, source_rtt]

    sentences_df = pd.DataFrame(products_list).transpose()
    sentences_df.columns = ['id', 'source', 'sentence_1', 'sentence_2', 'label', 'binary-label', 'source_rtt']

    return sentences_df

# KoEDA 증강
eda = EDA(morpheme_analyzer="Okt", alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, prob_rd=0.1)
model_name = 'klue/roberta-large'
tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=160)

def augment_df(df, max_len):
    augmented_data = []

    while len(augmented_data) < max_len:
        for i, row in tqdm(df.iterrows()):
            augmented_data.append([row['id'], row['sentence_1'], row['sentence_2'], row['label'], row['source'], row['binary-label'], row['source_rtt']])
            augmented_sentence = eda(row['sentence_1'])
            try:
                tokenizer(augmented_sentence, add_special_tokens=True, padding='max_length', truncation=True)
                augmented_data.append([row['id'], augmented_sentence, row['sentence_2'], row['label'], row['source'], row['binary-label'], row['source_rtt']])
            except:
                continue
    augmented_df = pd.DataFrame(augmented_data[:max_len], columns=['id', 'sentence_1', 'sentence_2', 'label', 'source', 'binary-label', 'source_rtt'])
    return augmented_df

def augment_data(df):
    # label을 기준으로 데이터를 나눕니다
    #df_00 = df[df['label'] < 0.5]
    df_01 = df[df['label'] == 0.1]
    df_02 =df[df['label'] == 0.2]
    #df_03 =df[df['label'] == 0.3]
    df_04 =df[df['label'] == 0.4]
    df_05 =df[df['label'] == 0.5]
    df_06 =df[df['label'] == 0.6]
    #df_07 =df[df['label'] == 0.7]
    df_08 =df[df['label'] == 0.8]
    #df_09 =df[df['label'] == 0.9]
    df_10 =df[df['label'] == 1.0]
    #df_11 =df[df['label'] == 1.1]
    df_12 =df[df['label'] == 1.2]
    #df_13 =df[df['label'] == 1.3]
    df_14 =df[df['label'] == 1.4]
    df_15 =df[df['label'] == 1.5]
    df_16 =df[df['label'] == 1.6]
    #df_17 =df[df['label'] == 1.7]
    df_18 =df[df['label'] == 1.8]
    #df_19 =df[df['label'] == 1.9]
    df_20 =df[df['label'] == 2.0]
    #df_21 =df[df['label'] == 2.1]
    df_22 =df[df['label'] == 2.2]
    #df_23 =df[df['label'] == 2.3]
    df_24 =df[df['label'] == 2.4]
    df_25 =df[df['label'] == 2.5]
    df_26 =df[df['label'] == 2.6]
    #df_27 =df[df['label'] == 2.7]
    df_28 =df[df['label'] == 2.8]
    #df_29 =df[df['label'] == 2.9]
    df_30 =df[df['label'] == 3.0]
    #df_31 =df[df['label'] == 3.1]
    df_32 =df[df['label'] == 3.2]
    #df_33 =df[df['label'] == 3.3]
    df_34 =df[df['label'] == 3.4]
    df_35 =df[df['label'] == 3.5]
    df_36 =df[df['label'] == 3.6]
    #df_37 =df[df['label'] == 3.7]
    df_38 =df[df['label'] == 3.8]
    #df_39 =df[df['label'] == 3.9]
    df_40 =df[df['label'] == 4.0]
    #df_41 =df[df['label'] == 4.1]
    df_42 =df[df['label'] == 4.2]
    #df_43 =df[df['label'] == 4.3]
    df_44 =df[df['label'] == 4.4]
    df_45 =df[df['label'] == 4.5]
    df_46 =df[df['label'] == 4.6]
    #df_47 =df[df['label'] == 4.7]
    df_48 =df[df['label'] == 4.8]
    #df_49 =df[df['label'] == 4.9]
    #df_50 = df[df['label'] <= 5]

    # 데이터를 배열에 저장합니다.
    df_list = [df_01, df_02, df_04, df_05, df_06, df_08, df_10, df_12, df_14, df_15, df_16, df_18,
            df_20, df_22, df_24, df_25, df_26, df_28, df_30, df_32, df_34, df_35, df_36, df_38,
            df_40, df_42, df_44, df_45, df_46, df_48]

    max_len = 1200 # 각 label별 1200개 만큼 증강
    
    augmented_df = pd.DataFrame()
    for idx, d in enumerate(df_list):
        if idx == 0:
            augmented_df = pd.concat([d])
            continue
        aug_df = augment_df(d, max_len)
        augmented_df = pd.concat([augmented_df, aug_df])

    augmented_df.reset_index(drop=True, inplace=True)
    return augmented_df

# label 빈도
def count_label(df):
    label_counts = df.groupby('label').agg({'label':['count']})
    label_counts['percent'] = round(label_counts[('label', 'count')]/sum(label_counts[('label', 'count')])*100, 1)

    print(label_counts)

# csv 파일 생성
def make_csv(df, data):
    # koeda_okt_0.1_aug: koeda에서 설정한 tokenizer(okt), 알파값(0.1)
    # ssodcs: spell-check, stop-word-removal, outlier-removal, down-sampling, copy-sentence, swap-sentence
    df.to_csv(f'../must_not_upload/koeda_ssodcs/koeda_okt_0.1_aug_ssodcs_{data}.csv')
    
## 실행
# 데이터 불러오기
train_df = make_df('train')

# 불용어 제거
train_df = remove_stopword(train_df, 'sentence_1', 'sentence_2')

# 토크나이징 & 문장 길이 차이 추가
tokenized_train_df = tokenizing(train_df)

# 문장 길이 이상치 제거
filtered_tokenized_train_df, outlier_tokenized_train_df  = remove_outliers(tokenized_train_df, '1_len - 2_len')
# 이상치 제거 확인
count_label(tokenized_train_df) # 원본 데이터 label별 빈도
count_label(filtered_tokenized_train_df) # 이상치 제거된 데이터 label별 빈도
count_label(outlier_tokenized_train_df) # 이상치 데이터 label별 빈도

# new_train_df 선언: 이상치 제거된 데이터에서 문장 길이 차이 열 제외하고 가져오기
new_train_df = filtered_tokenized_train_df[['id', 'source', 'sentence_1', 'sentence_2', 'label', 'binary-label', 'source_rtt']]

# copy sentence 1: 이상치로 제거된 데이터 중 label 0인 경우 sentence1 그대로 sentence2 복사, label은 5
filtered_sentences_df = outlier_tokenized_train_df[outlier_tokenized_train_df['label'] == 0.0]
filtered_sentences_copy_df = copy_sentence(filtered_sentences_df)

# new_train_df에 concat
new_train_df = pd.concat([new_train_df, filtered_sentences_copy_df])

# down sampling + copy sentence 2: source가 sampled이면서 label 0인 경우 down sampling하고, down sampling으로 제외된 문장들 모두 copy sentence
filtered_tokenized_train_setences_copy_df = down_copy_sentence(new_train_df)
filtered_tokenized_train_setences_copy_df

# copy한 문장 1200개 맞춰서 랜덤하게 가져옴.
df_5 = new_train_df[new_train_df['label'] == 5]
pct = (1200-len(df_5)) / len(filtered_tokenized_train_setences_copy_df)
sampled_set = filtered_tokenized_train_setences_copy_df.apply(sampling_func, sample_pct=pct)
sampled_set

new_train_df = pd.concat([new_train_df, sampled_set])

# label별 빈도 살펴보기
# label 0, 5는 1200으로 맞춰짐. 나머지 label들 중 빈도가 2자리 수로 매우 낮은 경우 존재함.
count_label(new_train_df)

# label 빈도가 2자리 수인 경우 swap sentence로 더블업
new_train_swap_sentences = new_train_df[(new_train_df['label'] == '0.5') | (new_train_df['label'] == '1.5') | (new_train_df['label'] == '2.5')|(new_train_df['label'] == '3.5')| (new_train_df['label'] == '4.5')|(new_train_df['label'] == '4.6')|(new_train_df['label'] == '4.8')]
new_train_swap_df = swap_sentence(new_train_swap_sentences)

new_train_df = pd.concat([new_train_df, new_train_swap_df])

# original
koeda_augmented_new_train_df = augment_data(new_train_df)
left_train_df = new_train_df[(new_train_df['label'] == 0.0) | (new_train_df['label'] == 5.0)]
final_train_df = pd.concat([left_train_df, koeda_augmented_new_train_df])

# koeda 과정에서 sentence 1,2에 none 발생한 경우 제외
drop_none(final_train_df)
final_train_df

# ssodcs: spell-check, stop-word-removal, outlier-removal, down-sampling, copy-sentence, swap-sentence
make_csv(final_train_df, 'train')
