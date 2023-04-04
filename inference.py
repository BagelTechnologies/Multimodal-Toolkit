import pickle
import pandas as pd
import numpy as np
import json
import torch
from torch import nn
from sklearn import metrics
from transformers import AutoTokenizer, AutoConfig
import importlib

multimodal_toolkit = importlib.import_module("Multimodal-Toolkit.multimodal_exp_args")
multimodal_transformers_data = importlib.import_module("Multimodal-Toolkit.multimodal_transformers.data")
multimodal_transformers_model = importlib.import_module("Multimodal-Toolkit.multimodal_transformers.model")


class MultimodalLoader:
    def __init__(self, multimodal_config_path, model_path, cat_num_transformers_path):
        with open(multimodal_config_path, 'r') as j:
            multimodal_config = json.loads(j.read())

        with open(cat_num_transformers_path, 'rb') as f:
            [labels_list, text_cols,
             cat_cols, cat_feat_dim, cat_transformer,
             numerical_cols, numerical_feat_dim, numerical_transformer] = pickle.load(f)

        self.text_cols = text_cols
        self.cat_cols = cat_cols
        self.numerical_cols = numerical_cols
        self.cat_transformer = cat_transformer
        self.numerical_transformer = numerical_transformer
        self.labels_list = labels_list

        self.label_col = multimodal_config["DATA-AGRS"]["LABEL-COL"]
        column_info_dict = {
            'text_cols': self.text_cols,
            'num_cols': self.numerical_cols,
            'cat_cols': self.cat_cols,
            'label_col': self.label_col,
            'label_list': self.labels_list
        }
        data_path = multimodal_config["DATA-AGRS"]["DATA-SPLIT-PATH"]
        data_args = multimodal_toolkit.MultimodalDataTrainingArguments(
            data_path=data_path,
            combine_feat_method=multimodal_config["MULTIMODAL-ARGS"]["COMBINE-FEAT-METHOD"],
            column_info=column_info_dict,
            task='classification'
        )

        model_name = multimodal_config["MODEL-ARGS"]["MODEL-NAME"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        tabular_config = multimodal_transformers_model.TabularConfig(num_labels=len(self.labels_list),
                                                                      cat_feat_dim=cat_feat_dim,
                                                                      numerical_feat_dim=numerical_feat_dim,
                                                                      **vars(data_args))
        config.tabular_config = tabular_config
        self.model = multimodal_transformers_model.AutoModelWithTabular.from_pretrained(model_name, config=config)

        self.model.load_state_dict(
            torch.load(model_path, map_location=None if torch.cuda.is_available() else torch.device('cpu')))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        if torch.cuda.is_available():
            self.model = self.model.to(self.device)

    def infer(self, df):
        text_list = list(df[self.text_cols].agg(f' {self.tokenizer.sep_token} '.join, axis=1))
        text_inputs = self.tokenizer(text_list, padding=True, truncation=True)
        cat_feats = self.cat_transformer.transform(df[self.cat_cols].values) \
            if self.cat_cols is not None and len(self.cat_cols) > 0 else None
        numerical_feats = self.numerical_transformer.transform(df[self.numerical_cols].values) \
            if self.numerical_cols is not None and len(self.numerical_cols) > 0 else None
        dataset = multimodal_transformers_data.TorchTabularTextDataset(text_inputs, cat_feats, numerical_feats)
        dataloader = torch.utils.data.DataLoader(dataset, 64)
        model_output = []
        with torch.no_grad():
            for model_inputs in dataloader:
                model_inputs = {key: val.to(self.device) for key, val in model_inputs.items()}
                _, logits, classifier_outputs = self.model(**model_inputs)
                model_output.append(logits)
        model_output = torch.cat(model_output, dim=0).to("cpu")
        return model_output

    def get_labels_list(self):
        return self.labels_list


if __name__ == '__main__':
    config_path = 'config.json'
    with open(config_path, 'r') as j:
        config = json.loads(j.read())

    data_path = config["DATA-AGRS"]["DATA-SPLIT-PATH"]
    test_df = pd.read_pickle(data_path + "test.gz")
    model_path = "logs/model_name/checkpoint-100/pytorch_model.bin"
    cat_num_transformers_path = data_path + config["DATA-AGRS"]["CAT-NUM-PREPROCESS-FILENAME"]
    mm_loader = MultimodalLoader(config_path, model_path, cat_num_transformers_path)
    model_output = mm_loader.infer(test_df)

    labels_np = np.array(mm_loader.get_labels_list())
    target_labels = list(labels_np[list(test_df[config["DATA-AGRS"]["LABEL-COL"]])])
    pred_labels = list(labels_np[model_output.argmax(axis=1)])
    print(metrics.classification_report(target_labels, pred_labels, digits=3, zero_division=1))
    
