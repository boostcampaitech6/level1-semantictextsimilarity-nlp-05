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


# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[], binary_targets=[]):
        self.inputs = inputs
        self.targets = targets
        self.binary_targets = binary_targets

    def __getitem__(self, idx):
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx]), torch.tensor(self.binary_targets[idx])

    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=140)

        self.target_columns = ['label']
        self.binary_target_columns = ['binary-label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        try:
            targets = data[self.target_columns].values.tolist()
            binary_targets = data[self.binary_target_columns].values.tolist()
        except:
            targets = []
            binary_targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets, binary_targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_targets, train_binary_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets, val_binary_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets, train_binary_targets)
            self.val_dataset = Dataset(val_inputs, val_targets, val_binary_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets, test_binary_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets, test_binary_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets, predict_binary_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [], [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=args.shuffle)

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
        self.second_loss_func = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y, y_binary = batch
        logits = self(x)
        loss = self.first_loss_func(logits, y.float())# + (0.5 * self.second_loss_func(logits, y_binary.float()))
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, y_binary = batch
        logits = self(x)
        loss = self.first_loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y, y_binary = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        rescaled_logits = logits

        return rescaled_logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.ws)
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

    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)

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

        dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                                args.test_path, args.predict_path)
        model = Model(args.model_name, config.lr, config.ws)
        wandb_logger = WandbLogger(project='klue-roberta-large')

        trainer = pl.Trainer(max_epochs=args.max_epoch, logger=wandb_logger, log_every_n_steps=1, precision=16)
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)

        if args.model_save == True:
            torch.save(model, f"klue-roberta-large-stopword_lr{config.lr}_weight_decay{args.weight_decay}_epoch{args.max_epoch}_model.pt")


    # Sweep 생성

    if args.mode == 'train':
        sweep_id = wandb.sweep(
            sweep=sweep_config,  # config 딕셔너리를 추가합니다.
            project='klue-roberta-large'  # project의 이름을 추가합니다.
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


    # if args.mode == 'train':
    #     existed_model = glob('model.pt')
    #     if len(existed_model) > 0:
    #         print("model.pt 가 존재합니다. 덮어쓰지 않도록 model.pt의 이름을 바꿔주세요.")
    #     else:
    #         model = Model(args.model_name, args.learning_rate, args.weight_decay)
    #
    #         wandb_logger = WandbLogger(project="klue-sts")
    #         trainer = pl.Trainer(
    #             accelerator="gpu",
    #             devices=1,
    #             logger=wandb_logger,
    #             max_epochs=args.max_epoch,
    #             log_every_n_steps=1,
    #         )
    #
    #         trainer.fit(model=model, datamodule=dataloader)
    #         trainer.test(model=model, datamodule=dataloader)
    #
    #         # 학습이 완료된 모델을 저장합니다.
    #         torch.save(model, 'model.pt')
    #
    # elif args.mode == 'test':
    #     model = torch.load('model.pt')
    #     predictions = trainer.predict(model=model, datamodule=dataloader)
    #     predictions = list(round(float(i), 1) for i in torch.cat(predictions))
    #
    #     output = pd.read_csv('../data/sample_submission.csv')
    #     output['target'] = predictions
    #     output.to_csv('output.csv', index=False)
