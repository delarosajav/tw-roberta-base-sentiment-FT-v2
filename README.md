---
datasets:
- Sp1786/multiclass-sentiment-analysis-dataset
language:
- en
metrics:
- accuracy
- precision
- recall
- f1
base_model:
- cardiffnlp/twitter-roberta-base-sentiment
pipeline_tag: text-classification
library_name: transformers
tags:
- roBERTa
- text-classification
- sentiment-analysis
- english
- fine-tuned
- nlp
- transformers
- content-moderation
- social-media-analysis
---


# tw-roberta-base-sentiment-FT-v2

This model is a second fine-tuned version of [cardiffnlp/twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment), trained on the [Sp1786/multiclass-sentiment-analysis-dataset](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset). It builds on the first iteration by incorporating optimized techniques. Specifically, the dataset proportions were adjusted to refine the division of the training, evaluation, and test sets, leading to a more balanced and representative fine-tuning process. Compared to the initial model, this version demonstrates improved performance, with enhanced accuracy and robustness for the task.

**It is specifically fine-tuned to analyze user-generated content such as opinions, reviews, comments, and general customer feedback. It is designed for sentiment analysis in the context of understanding public perception, trend analysis, and gathering insights into consumer satisfaction.**

## Try it out

You can interact with the model directly through the [Inference Endpoint](https://huggingface.co/spaces/delarosajav95/tw-roberta-base-sentiment-FT-v2):

[![Open Inference Endpoint](https://img.shields.io/badge/Open_Inference_Endpoint-blue)](https://huggingface.co/spaces/delarosajav95/tw-roberta-base-sentiment-FT-v2)

## Full classification example in Pyhton:

```python
from transformers import pipeline

pipe = pipeline(model="delarosajav95/tw-roberta-base-sentiment-FT-v2")

inputs = ["The flat is very nice but it's too expensive and the location is very bad.",
  "I loved the music, but the crowd was too rowdy to enjoy it properly.",
  "They believe that I'm stupid and I like waiting for hours in line to buy a simple coffee."
]

result = pipe(inputs, return_all_scores=True)

label_mapping = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
for i, predictions in enumerate(result):
  print("==================================")
  print(f"Text {i + 1}: {inputs[i]}")
  for pred in predictions:
    label = label_mapping.get(pred['label'], pred['label'])
    score = pred['score']
    print(f"{label}: {score:.2%}")
```

Output:

```pyhton
==================================
Text 1: The flat is very nice but it's too expensive and the location is very bad.
Negative: 78.54%
Neutral: 20.66%
Positive: 0.80%
==================================
Text 2: I loved the music, but the crowd was too rowdy to enjoy it properly.
Negative: 5.18%
Neutral: 93.34%
Positive: 1.48%
==================================
Text 3: They believe that I'm stupid and I like waiting for hours in line to buy a simple coffee.
Negative: 82.37%
Neutral: 16.85%
Positive: 0.79%
```

## Pipeline API:

```pyhton
from transformers import pipeline

url = "delarosajav95/tw-roberta-base-sentiment-FT-v2"

classifier = pipeline("sentiment-analysis", model=url)

text = "text to classify"

result = classifier(text, return_all_scores=True)

label_mapping = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
for i, predictions in enumerate(result):
  print("==================================")
  print(f"Text {i + 1}: {text}")
  for pred in predictions:
    label = label_mapping.get(pred['label'], pred['label'])
    score = pred['score']
    print(f"{label}: {score:.2%}")
```

## Metrics and results:

It achieves the following results on the *evaluation set* (last epoch):
- 'eval_loss': 0.8515534996986389
- 'eval_accuracy': 0.7709153779656133
- 'eval_precision_per_label': [0.7665824384080859, 0.7291611185086552, 0.8197707736389684]
- 'eval_recall_per_label': [0.7988808426596445, 0.695630081300813, 0.8324119871981379]
- 'eval_f1_per_label': [0.7823984526112185, 0.7120010401768301, 0.8260430200664068]
- 'eval_precision_weighted': 0.7699940216435469
- 'eval_recall_weighted': 0.7709153779656133
- 'eval_f1_weighted': 0.7701923401341971
- 'eval_runtime': 47.0811
- 'eval_samples_per_second': 221.129
- 'eval_steps_per_second': 27.654
- 'epoch': 4.0

It achieves the following results on the *test set*:
- 'eval_loss': 0.8580234050750732
- 'eval_accuracy': 0.7714916914801652
- 'eval_precision_per_label': [0.7692307692307693, 0.7117024024799793, 0.8409554325662686]
- 'eval_recall_per_label': [0.7787552948843272, 0.7161424486612945, 0.8260371959942775]
- 'eval_f1_per_label': [0.7739637305699482, 0.713915522155999, 0.8334295612009238]
- 'eval_precision_weighted': 0.7720514465400845
- 'eval_recall_weighted': 0.7714916914801652
- 'eval_f1_weighted': 0.7717379713044402

## Training Details and Procedure

### Main Hyperparameters:

The following hyperparameters were used during training:
- evaluation_strategy: "epoch"
- learning_rate: 1e-5
- per_device_train_batch_size: 8
- per_device_eval_batch_size: 8
- num_train_epochs: 4
- optimizer: AdamW
- weight_decay: 0.01
- save_strategy: "epoch"
- lr_scheduler_type: "linear"
- warmup_steps: 820
- logging_steps: 10


#### Preprocessing and Postprocessing:

- Needed to manually map dataset creating the different sets: train 50%, validation 25%, and test 25%.
- Seed=123
- Num labels = 3 | srt("negative", "neutral", "positive") int(0, 1, 2)
- Dynamic Padding through DataCollator was used.

### Framework versions

- Transformers 4.47.0
- Pytorch 2.5.1+cu121
- Datasets 3.2.0
- Tokenizers 0.21.0

## CITATION:

If you use this model, please cite the following paper:

```bibitex
@inproceedings{barbieri-etal-2020-tweeteval,
    title = "{T}weet{E}val: Unified Benchmark and Comparative Evaluation for Tweet Classification",
    author = "Barbieri, Francesco  and
      Camacho-Collados, Jose  and
      Espinosa Anke, Luis  and
      Neves, Leonardo",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.148",
    doi = "10.18653/v1/2020.findings-emnlp.148",
    pages = "1644--1650"
}
```

## More Information

- Fine-tuned by Javier de la Rosa Sánchez.
- javier.delarosa95@gmail.com
- https://www.linkedin.com/in/delarosajav95/