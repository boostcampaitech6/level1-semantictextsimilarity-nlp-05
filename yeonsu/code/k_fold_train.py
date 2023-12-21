import argparse
import random

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

import os
import requests
import json
from pytorch_lightning.loggers import WandbLogger
from transformers import get_linear_schedule_with_warmup

from glob import glob
import wandb
from sklearn.model_selection import KFold


# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 if문을, 없다면 else문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    # 'return 100'이면 1에폭에 100개의 데이터만 사용합니다
    def __len__(self):
        return len(self.inputs)


class KfoldDataloader(pl.LightningDataModule):
    def __init__(self,
                 model_name,
                 batch_size,
                 shuffle,
                 bce,
                 train_path,
                 dev_path,
                 test_path,
                 predict_path,
                 k = 1,  # fold number
                 split_seed = 12345,  # split needs to be always the same for correct cross validation
                 num_splits = 10,
                 ):

        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bce = bce
        self.k = k
        self.split_seed = split_seed
        self.num_splits = num_splits

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=120)
        # self.set_preprocessing()

        self.target_columns = ['label']
        self.binary_target_columns = ['binary-label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']


    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])

        return data


    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 데이터 준비
            # original
            # total_data = self.read_json('train')

            # K fold
            total_data = pd.read_csv(self.train_path)

            total_input, total_targets = self.preprocessing(total_data)
            total_dataset = Dataset(total_input, total_targets)

            # 데이터셋 num_splits 번 fold
            kf = KFold(n_splits=self.num_splits, shuffle=self.shuffle, random_state=self.split_seed)
            all_splits = [k for k in kf.split(total_dataset)]
            # k번째 fold 된 데이터셋의 index 선택
            train_indexes, val_indexes = all_splits[self.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            # fold한 index에 따라 데이터셋 분할
            self.train_dataset = [total_dataset[x] for x in train_indexes]
            self.val_dataset = [total_dataset[x] for x in val_indexes]

        else:
            # 평가데이터 준비
            # test_data = self.read_json('dev')

            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)
            self.predict_dataset = Dataset(test_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)


class Model(pl.LightningModule):
    def __init__(self, model_name, lr, wd=0, ws=0):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.wd = wd
        self.ws = ws

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name,
            num_labels=1,
        )

        # for param in self.plm.parameters():
        #      print(param.requires_grad)

        # Loss 계산을 위해 사용될 MSELoss를 호출합니다.
        self.first_loss_func = torch.nn.MSELoss()
        self.evaluation = torchmetrics.PearsonCorrCoef()

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.first_loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.first_loss_func(logits, y.float())
        self.log("val_loss", loss)
        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return self.evaluation(logits, y)

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=5, type=int)
    parser.add_argument('--warmup_steps', default=297, type=int)
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--bce', default=False)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1.9652545776142725e-05, type=float)
    parser.add_argument('--train_path', default='../data/rtt_swap_stopword/train.csv')
    parser.add_argument('--dev_path', default='../data/rtt_swap_stopword/dev.csv')
    parser.add_argument('--test_path', default='../data/rtt_swap_stopword/dev.csv')
    parser.add_argument('--predict_path', default='../data/rtt_swap_stopword/test.csv')
    parser.add_argument('--mode', required=True)
    parser.add_argument('--model_save', default=True)
    parser.add_argument('--wandb_name', default='project', required=True, type=str)
    args = parser.parse_args()

    sweep_config = {
        'method': 'random',  # random: 임의의 값의 parameter 세트를 선택
        "parameters": {
            "batch_size": {"values": [16]},
            "lr" : {"values" : [2.306e-06]},
            # "lr": {"max": 2e-5, "min": 1.9e-5},
            "ws": {"max": 300, "min": 0}
        },
    }

    if args.bce:
        sweep_config['metric'] = {  # sweep_config의 metric은 최적화를 진행할 목표를 설정합니다.
            'name': 'val_f1',  # F1 점수가 최대화가 되는 방향으로 학습을 진행합니다.
            'goal': 'maximize'
        }
    else:
        sweep_config['metric'] = {'name': 'val_pearson', 'goal': 'maximize'}  # pearson 점수가 최대화가 되는 방향으로 학습을 진행합니다.


    def sweep_train(config=None):
        wandb.init(config=config)
        config = wandb.config
        print("----------------lr----------------")
        print(config.lr)
        print(config.ws)
        print("----------------------------------")

        Kmodel = Model(args.model_name, config.lr)
        wandb_logger = WandbLogger(project='klue-roberta-large-kfold')

        results = []
        # K fold 횟수 3
        nums_folds = 5
        split_seed = 12345

        for k in range(nums_folds):
            kfdataloader = KfoldDataloader(args.model_name, args.batch_size,
                                           args.shuffle, args.bce,
                                           args.train_path, args.dev_path, args.test_path, args.predict_path,
                                           k=k, split_seed=split_seed, num_splits=nums_folds)
            kfdataloader.prepare_data()
            kfdataloader.setup()

            trainer = pl.Trainer(max_epochs=3, logger=wandb_logger, log_every_n_steps=1, precision=16)
            trainer.fit(model=Kmodel, datamodule=kfdataloader)
            score = trainer.test(model=Kmodel, datamodule=kfdataloader)

            results.extend(score)

        if args.model_save == True:
            torch.save(model, f"klue-roberta-large-stopword_lr{config.lr}_kfold_model.pt")


    if args.mode == 'train':
        sweep_id = wandb.sweep(
            sweep=sweep_config,  # config 딕셔너리를 추가합니다.
            project='klue-roberta-large-kfold'  # project의 이름을 추가합니다.
        )
        wandb.agent(
            sweep_id=sweep_id,  # sweep의 정보를 입력하고
            function=sweep_train,  # train이라는 모델을 학습하는 코드를
            count=1  # 총 3회 실행해봅니다.
        )

    elif args.mode == 'test':
        pt_name = 'klue-roberta-large_lr2.306e-06_epoch5_model.pt'
        trainer = pl.Trainer(max_epochs=args.max_epoch, log_every_n_steps=1, precision=16)

        model = torch.load(pt_name)
        predictions = trainer.predict(model=model, datamodule=dataloader)
        predictions = list(round(float(i), 1) for i in torch.cat(predictions))

        output = pd.read_csv('../data/sample_submission.csv')
        output['target'] = predictions
        output.to_csv('./outputs/' + pt_name[:-3] + '.csv', index=False)
