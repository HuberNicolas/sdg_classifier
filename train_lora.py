from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction, DataCollatorWithPadding
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import json


# https://www.youtube.com/watch?v=Us5ZFp16PaU
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb
# https://discuss.huggingface.co/t/how-to-save-my-model-to-use-it-later/20568
# https://datascience.stackexchange.com/questions/45174/how-to-use-sklearn-train-test-split-to-stratify-data-for-multi-label-classificat


# Load the dataset splits from disk
dataset = DatasetDict.load_from_disk("./data/preprocessed/sdg_dataset_splits_multilabel")


example = dataset["train"][0]
print(example)

labels = [label for label in dataset["train"].features.keys() if label not in ['abstract']]
print(labels)
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
print(id2label)
print(label2id)


model_path = 'distilbert/distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_path)


def preprocess_data(examples):
  # take a batch of texts
  text = examples["abstract"]
  # encode them
  encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
  # add labels
  labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
  # create numpy array of shape (batch_size, num_labels)
  labels_matrix = np.zeros((len(text), len(labels)))
  # fill numpy array
  for idx, label in enumerate(labels):
    labels_matrix[:, idx] = labels_batch[label]

  encoding["labels"] = labels_matrix.tolist()
  
  return encoding

encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset["train"].column_names)
print(encoded_dataset)


example = encoded_dataset['train'][0]
print(example.keys())

print(tokenizer.decode(example['input_ids']))
print(example['labels'])

print([id2label[idx] for idx, label in enumerate(example['labels']) if label == 1.0])

encoded_dataset.set_format("torch")
     

model = AutoModelForSequenceClassification.from_pretrained(model_path, 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id,
                                                           torch_dtype=torch.float32) # https://stackoverflow.com/questions/75641074/i-run-stable-diffusion-its-wrong-runtimeerror-layernormkernelimpl-not-implem


batch_size = 8
metric_name = "f1"
     


args = TrainingArguments(
    f"bert-finetuned-sdg-english",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    #push_to_hub=True,
)


    
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    try:
        roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    except ValueError:
        roc_auc = 0.0  # or handle appropriately
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result


print(encoded_dataset["train"][0]['labels'].type())

print(encoded_dataset["train"]['input_ids'][0])


#forward pass
outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0), labels=encoded_dataset['train'][0]['labels'].unsqueeze(0))
print(outputs)




###
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


print(model)

for name, module in model.named_modules():
    print(name)


# Class imbalance -> Use label weights during training to handle class imbalance

from torch.nn import BCEWithLogitsLoss

# Calculate label weights
label_counts = torch.sum(encoded_dataset["train"]["labels"], axis=0)
label_counts = label_counts.numpy()
label_weights = 1.0 / (label_counts + 1e-6)  # Add epsilon to avoid division by zero
# Normalize
label_weights = label_weights / label_weights.sum() * len(label_weights)

label_sums = torch.sum(encoded_dataset["train"]["labels"], dim=0).numpy()
print("Label counts:", label_sums)
print("Total samples:", len(encoded_dataset["train"]))
print("Class proportions:", label_sums / len(encoded_dataset["train"]))


# Custom loss function
# loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor(label_weights))

# Weighted loss
loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor(label_weights, dtype=torch.float32))



# https://arxiv.org/pdf/2106.09685
from peft import LoraConfig, get_peft_model
config = LoraConfig(
   r=16, # attention heads
   lora_alpha=32, # alpha scaling
   lora_dropout=0.05,
   # https://stackoverflow.com/questions/76768226/target-modules-for-applying-peft-lora-on-different-models
   # # query Responsible for computing attention scores by projecting input embeddings.
   # # value Determines how much of each token contributes to the attention output.
   target_modules=["attention.q_lin", "attention.v_lin"], 
   bias="none",
   task_type="SEQ_CLS",
)
model = get_peft_model(model, config)
print_trainable_parameters(model)



model.config.loss_fn = loss_fn


batch_size = 4
metric_name = "f1"

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

args = TrainingArguments(
        output_dir='bert-finetuned-lora-sdg-english',
        eval_strategy = "epoch",
        save_strategy = "epoch",
        per_device_train_batch_size=batch_size, 
        gradient_accumulation_steps=1,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        warmup_steps=100, 
        learning_rate=2e-6, 
        #fp16=True, # requires GPU
        logging_steps=1,
)

trainer = Trainer(
    model=model, 
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    args=args,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# model.save_pretrained("./output/distilbert-sdg-classifier") cant save with custom loss
# tokenizer.save_pretrained("./output/distilbert-sdg-classifier")
metrics = trainer.evaluate()
print(metrics)

test_results = trainer.evaluate(encoded_dataset["test"])
print(test_results)