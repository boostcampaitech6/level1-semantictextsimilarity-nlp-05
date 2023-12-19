import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import Counter
from glob import glob
import torch.nn.functional as F
import torch


class Ensemble():
    def __init__(self):
        # self.files = glob('./outputs2/*.csv')
        self.files = os.listdir('./outputs2')
        self.files = [(file, float(file.replace('.csv', "").split('_')[1])) for file in self.files]

    def soft_vote_ensemble(self):
        print("===========================ensemble start===========================")
        num_files = len(self.files)  # (filename, score)
        print(self.files)
        scores = torch.Tensor([inference[1] for inference in self.files])
        inf_list = [pd.read_csv('./outputs2/' + inference[0])['target'] for inference in self.files]

        scores = F.softmax(scores, dim=-1)
        inf_list = [inf_list[i] * scores[i].item() for i in range(num_files)]
        concatenated_inf = pd.concat(inf_list, axis=1)
        ensemble_output = pd.Series(concatenated_inf.sum(axis=1))

        for i in range(len(ensemble_output)):
            if ensemble_output.iloc[i] > 5:
                ensemble_output.iloc[i] = 5
            elif ensemble_output.iloc[i] < 0:
                ensemble_output.iloc[i] = 0

        output = pd.read_csv('../data/sample_submission.csv')
        output['target'] = ensemble_output
        output.to_csv('[last3]ensemble_output.csv', index=False)
        print("============================ensemble end============================")


# e = Ensemble([('output.csv', 0.5), ('output.csv', 0.8), ('output.csv', 0.7)])
e = Ensemble()
e.soft_vote_ensemble()