import argparse
import random
import os
import re

import pandas as pd
from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import warnings


transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*TensorBoard support*")
warnings.filterwarnings("ignore", ".*target is close to zero*")


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


test_pearson = 0


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets


    def __getitem__(self, idx):
        input_ids, attention_mask, token_type_ids = self.inputs[idx]

        input_ids_tensor = torch.tensor(input_ids)
        attention_mask_tensor = torch.tensor(attention_mask)
        token_type_ids_tensor = torch.tensor(token_type_ids)

        if len(self.targets) > 0:
            target = self.targets[idx]
            return input_ids_tensor, attention_mask_tensor, token_type_ids_tensor, torch.tensor(target)
        else:
            return input_ids_tensor, attention_mask_tensor, token_type_ids_tensor


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

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=120)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            # text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            text = [item[text_column] for text_column in self.text_columns]
            outputs = self.tokenizer(text=text[0], text_pair=text[1], add_special_tokens=True, padding='max_length',
                                     truncation=True)

            input_ids = outputs['input_ids']
            attention_mask = outputs['attention_mask']
            token_type_ids = outputs['token_type_ids']

            data.append((input_ids, attention_mask, token_type_ids))
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
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=args.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)


class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)

        # # ElectraModel 부분의 모든 파라미터를 freeze # 학습시간 1분 30초로 단축
        # for param in self.plm.electra.parameters():
        #     param.requires_grad = False

        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        # self.loss_func = torch.nn.L1Loss()
        self.loss_func = torch.nn.MSELoss()

    # def forward(self, x):
    #     x = self.plm(x)['logits']
    #     return x
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, y = batch
        logits = self(input_ids, attention_mask, token_type_ids)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, y = batch
        logits = self(input_ids, attention_mask, token_type_ids)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)
        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
        return loss

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, y = batch
        logits = self(input_ids, attention_mask, token_type_ids)
        test_pearson = torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze())
        self.log("test_pearson", test_pearson)

    def on_test_epoch_end(self):
        aggregated_test_pearson = self.trainer.logged_metrics['test_pearson']
        print(f">>>>>>>>>>>>>> Aggregated Test Pearson: {aggregated_test_pearson}")
        self.test_pearson_value = aggregated_test_pearson.item()

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids = batch
        logits = self(input_ids, attention_mask, token_type_ids)
        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)  # weight_decay=0.01
        return optimizer


if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=2.306e-06, type=float)
    parser.add_argument('--train_path', default='../must_not_upload/rtt_swap_stopword/train.csv')
    parser.add_argument('--dev_path', default='../must_not_upload/rtt_swap_stopword/dev.csv')
    parser.add_argument('--test_path', default='../must_not_upload/rtt_swap_stopword/dev.csv')
    parser.add_argument('--predict_path', default='../must_not_upload/rtt_swap_stopword/test.csv')
    parser.add_argument('--inference', default=False)
    args = parser.parse_args()

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)

    model = Model(args.model_name, args.learning_rate)
    wandb_logger = WandbLogger(project="klue-roberta-large")
    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=wandb_logger, max_epochs=args.max_epoch,
                         log_every_n_steps=1, precision=16)

    if args.inference == False:
        # Train part
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)

        model_save_path = '../models'
        model_subdirectory = args.model_name.replace('/', '_')
        test_pearson_value = model.test_pearson_value
        model_filename = f'{model_subdirectory}_{args.batch_size}_{args.max_epoch}_{test_pearson_value:.4f}.pt'
        full_path = os.path.join(model_save_path, model_filename)

        # 학습이 완료된 모델을 저장합니다.
        torch.save(model, full_path)
    else:
        # Inference part
        # 저장된 모델로 예측을 진행합니다.
        model_save_path = '../models'
        model_subdirectory = args.model_name.replace('/', '_')  # 특수 문자를 대체
        pattern = f'{model_subdirectory}_{args.batch_size}_{args.max_epoch}'

        # 모델 저장 경로에서 모든 파일 목록을 가져옴
        files = os.listdir(model_save_path)

        # 원하는 패턴으로 시작하는 파일 찾기
        model_files = [f for f in files if f.startswith(pattern)]

        if model_files:
            # 가장 최근 파일을 선택하거나 다른 기준으로 선택
            model_file = model_files[-1]  # 가장 마지막 파일 선택
            full_path = os.path.join(model_save_path, model_file)

            # 모델 로드
            model = torch.load(full_path)
        else:
            print("해당 패턴으로 시작하는 모델 파일이 없습니다.")

        # 저장 csv 파일
        output_directory = '../outputs'
        output_filename = f'{model_file[:-3]}_output.csv'  # 직접 파일명 생성
        full_output_path = os.path.join(output_directory, output_filename)

        # 모델 로드 후 predict
        model = torch.load(full_path)
        predictions = trainer.predict(model=model, datamodule=dataloader)

        # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
        predictions = list(round(float(i), 1) for i in torch.cat(predictions))

        # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
        output = pd.read_csv('../must_not_upload/data/sample_submission.csv')
        output['target'] = predictions
        output.to_csv(full_output_path, index=False)