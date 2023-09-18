import argparse
import logging
import os
import json
from sklearn import metrics

import numpy as np
import pandas as pd
import pickle
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    EvalPrediction,
    set_seed
)
from transformers.training_args import TrainingArguments

from multimodal_exp_args import ModelArguments, MultimodalDataTrainingArguments
from multimodal_transformers.data import load_data_from_folder
from multimodal_transformers.model import TabularConfig
from multimodal_transformers.model import AutoModelWithTabular
from evaluation import calc_classification_metrics
from util import replace_col_wildcards

parser = argparse.ArgumentParser(description='Text BERT Classification')
parser.add_argument('-s', dest='skip_pkl_split_data', action='store_true', help="skip saving splitted data")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
os.environ['COMET_MODE'] = 'DISABLED'

with open('config.json', 'r') as j:
    config = json.loads(j.read())

BEST_MODEL_DIR = os.path.join(config["TRAINING-ARGS"]["OUTPUT-DIR"], "best_model")
label_col = config["DATA-AGRS"]["LABEL-COL"]
data_df = pd.read_pickle(config["DATA-AGRS"]["DATA-DF-PATH"])
labels_list = list(np.sort(data_df[label_col].unique()))

labels = dict()
for idx, val in enumerate(labels_list):
    labels[val] = idx

data_df[label_col] = data_df[label_col].apply(lambda label_val: labels[label_val])
split_fracs = config["DATA-AGRS"]["TRAIN-VAL-SPLIT-FRACTIONS"]
train_df, val_df, test_df = np.split(data_df.sample(frac=1), [int(split_fracs[0] * len(data_df)), int(split_fracs[1] * len(data_df))])
print(f'Data labels num: {len(labels_list)}')
print('Num examples train-val-test')
print(len(train_df), len(val_df), len(test_df))

data_path = config["DATA-AGRS"]["DATA-SPLIT-PATH"]
if not args.skip_pkl_split_data:
    train_df.to_pickle(data_path + 'train.gz')
    val_df.to_pickle(data_path + 'val.gz')
    test_df.to_pickle(data_path + 'test.gz')

freezing_layers = config['TRAINING-ARGS']['FREEZE-LAYERS']
text_cols = config["DATA-AGRS"]["TEXT-COLS"]
cat_cols = config["DATA-AGRS"]["CAT-COLS"]
numerical_cols = config["DATA-AGRS"]["NUMERICAL-COLS"]
numerical_cols = replace_col_wildcards(numerical_cols, data_df.columns)

column_info_dict = {
    'text_cols': text_cols,
    'num_cols': numerical_cols,
    'cat_cols': cat_cols,
    'label_col': label_col,
    'label_list': labels_list
}

model_args = ModelArguments(
    model_name_or_path=config["MODEL-ARGS"]["MODEL-NAME"]
)

data_args = MultimodalDataTrainingArguments(
    data_path=data_path,
    combine_feat_method=config["MULTIMODAL-ARGS"]["COMBINE-FEAT-METHOD"],
    column_info=column_info_dict,
    task='classification'
)

training_args = TrainingArguments(
    output_dir=config["TRAINING-ARGS"]["OUTPUT-DIR"],
    logging_dir=config["TRAINING-ARGS"]["LOGGING-DIR"],
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=config["TRAINING-ARGS"]["PER-DEVICE-BATCH-SIZE"],
    num_train_epochs=config["TRAINING-ARGS"]["EPOCH-SIZE"],
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="epoch",
    metric_for_best_model="acc",
    greater_is_better=True,
    load_best_model_at_end=True,
    report_to=["none"]
)

set_seed(training_args.seed)

tokenizer_path_or_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
print('Specified tokenizer: ', tokenizer_path_or_name)
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path_or_name,
    cache_dir=model_args.cache_dir,
)

# Get Datasets
train_dataset, val_dataset, test_dataset, cat_transformer, numerical_transformer = load_data_from_folder(
    data_args.data_path,
    data_args.column_info['text_cols'],
    tokenizer,
    label_col=data_args.column_info['label_col'],
    label_list=data_args.column_info['label_list'],
    categorical_cols=data_args.column_info['cat_cols'],
    numerical_cols=data_args.column_info['num_cols'],
    sep_text_token_str=tokenizer.sep_token,
)

cat_feat_dim = train_dataset.cat_feats.shape[1] if train_dataset.cat_feats is not None else 0
numerical_feat_dim = train_dataset.numerical_feats.shape[1] if train_dataset.numerical_feats is not None else 0
ohe = cat_transformer.get_ohe() if cat_transformer is not None else None
with open(data_path + config["DATA-AGRS"]["CAT-NUM-PREPROCESS-FILENAME"], 'wb') as f:
    pickle.dump([labels_list, text_cols,
                 cat_cols, cat_feat_dim, ohe,
                 numerical_cols, numerical_feat_dim, numerical_transformer], f, protocol=pickle.HIGHEST_PROTOCOL)

num_labels = len(np.unique(train_dataset.labels))
print(f'Training labels num: {num_labels}')
config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
)
tabular_config = TabularConfig(num_labels=num_labels,
                               cat_feat_dim=cat_feat_dim,
                               numerical_feat_dim=numerical_feat_dim,
                               **vars(data_args))
config.tabular_config = tabular_config

model = AutoModelWithTabular.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    config=config,
    cache_dir=model_args.cache_dir
)

if len(freezing_layers) > 0:
    print(f"freezing layers: {freezing_layers}")
    print("total number of trainable parameters prior freezing:" + str(sum(p.numel() for p in model.parameters()
                                                                           if p.requires_grad)))
    for name, param in model.named_parameters():
        if any([x in name for x in freezing_layers]):
            param.requires_grad = False
    print("total number of trainable parameters after freezing:" + str(sum(p.numel() for p in model.parameters()
                                                                           if p.requires_grad)))


def calc_classification_metrics_fn(p: EvalPrediction):
    pred_labels = np.argmax(p.predictions[0], axis=1)
    result = calc_classification_metrics(p.predictions[0], p.label_ids)

    labels_np = np.array(labels_list)
    print(f'evaluation report:\n'
          f"{metrics.classification_report(list(labels_np[p.label_ids]), list(labels_np[pred_labels]), digits=3, zero_division=1)}")
    return result


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=calc_classification_metrics_fn,
)

trainer.train()

# Save the best model to a specified directory
trainer.save_model(BEST_MODEL_DIR)
