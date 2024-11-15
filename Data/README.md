### Dataset Description

The Twitter Financial News dataset is an English-language dataset containing an annotated corpus of finance-related tweets. This dataset is used to classify finance-related tweets for their sentiment.

The dataset holds 11,932 documents annotated with 3 labels:

```python
sentiments = {
    "LABEL_0": "Bearish", 
    "LABEL_1": "Bullish", 
    "LABEL_2": "Neutral"
}  
```

The data was collected using the Twitter API. The current dataset supports the multi-class classification task.

### Data Splits
There are 2 splits: train and validation. Below are the statistics:

| Dataset Split | Number of Instances in Split                |
| ------------- | ------------------------------------------- |
| Train         | 9,938                                       |
| Validation    | 2,486                                       |

### Licensing Information
The Twitter Financial Dataset (sentiment) version 1.0.0 is released under the MIT License.

### Dataset Source
The dataset used for this project is sourced from the [zeroshot](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) account on Hugging Face. It focuses on financial news sentiment analysis from tweets, making it highly relevant for understanding market sentiment. This dataset serves as a foundational component for training and fine-tuning the sentiment analysis models used in this project.
