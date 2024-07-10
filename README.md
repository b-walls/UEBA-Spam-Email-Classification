# Spam Classification Model

## Data Preparation

### `data_preparation.ipynb`

The `data_preparation.ipynb` notebook is dedicated to the preparation of data for training spam classification models. This notebook includes:

- **Feature Engineering**: Feature engineering process is documented within notebook.
- **Text Cleaning**: Comprehensive text cleaning procedures, including removing special characters, stop-word removal, and lemmatization, are implemented.
- **Sentiment Analysis**: Additionally, a pre-tuned sentiment analysis model was used to gain extra insights into the emails. This model is sourced from a [Hugging Face post](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) and provides sentiment scores for the emails.

Given the computational complexity and the volume of semi-raw data, the data preparation tasks are separated from the model. This separation facilitates easier data management and storage, ensuring the data is ready for model training and testing.

## How to Use

1. **Data Preparation**: Run the `data_preparation.ipynb` notebook to clean and prepare the dataset.
2. **Model Training**: Run the `WIP` Used for the training of the model.
3. **Model Classification**: Run the `WIP`.

## Acknowledgements

- **Dataset**: The dataset used to train the spam classification models is derived from a GitHub repository, which compiled data from the study ["Spam Filtering with Naive Bayes - Which Naive Bayes?"](https://nes.aueb.gr/ipl/nlp/pubs/ceas2006_paper.pdf). You can access the dataset here: [Enron Spam Data](https://github.com/MWiechmann/enron_spam_data).
- **Sentiment Analysis Model**: [Hugging Face - CardiffNLP](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
