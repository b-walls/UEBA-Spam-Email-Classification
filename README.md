# Spam Classification Model

## Data

The `spam_data.zip` folder contains the original semi-raw data (`enron_spam_data.csv`) and the preprocessed data (`clean_enron_spam_data.pkl`) used to train the model. 

## Data Preparation

### `data_preparation.ipynb`

The `data_preparation.ipynb` notebook is dedicated to the preparation of data for training a spam classification ML model. This notebook includes:

- **Feature Engineering**: Feature engineering process is documented within notebook.
- **Text Cleaning**: Comprehensive text cleaning procedures, including removing special characters, stop-word removal, and lemmatization, are implemented.
- **Sentiment Analysis**: Additionally, a pre-tuned sentiment analysis model was used to gain extra insights into the emails. This model is sourced from a [Hugging Face post](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) and provides sentiment scores for the emails.
- **TF-IDF**: Term Frequency - Inverse Document Frequency is used to tokenize the data for the models. This approach allows the most important words be analyzed by the model
- **Export Data**: Once all preprocessing is complete there is an export of the data file to be imported by the model training notebook.

Given the computational complexity and the volume of semi-raw data, the data preparation tasks are separated from the model training and classification. This separation facilitates easier data management and storage, ensuring the data is ready for model training and testing when changes to the preprocessed data may be necessary.

## Model Training

### `model_training.ipynb`

The `model_training.ipynb` notebook is dedicated to the training of ML models to classify spam emails. This notebook includes:

- **Random Forest Classifier**: The first model is a Random Forest Classifier. After investigation and hyper parameter tuning it appears that the default parameters work best for the dataset reaching a 0.979 accuracy score.
- **XGBoost Classifier**: The second model is a XGBoost Classifier. After tuning the model to the data it reached an accuracy of 0.982, and better recall and precision scores than the Random Forest model.
- **Export Model**: After training there is a method called `export_model(model, filename, directory)` this is used to compress the model to a .pkl file and it's main use is to expand the capability of this project. The models can be imported into other python scripts.

## Model in Action

### `run_classification`

The `run_classification.py` script is dedicated to running a UI window that allows you to classify emails as either spam or not spam.

- **Step 1**: Run the file and click `File` > `Import`. This will open the file explorer in the script's directory. First, select the compressed model's .pkl file, then select the compressed training data.
- **Step 2**: Type in an email into the appropraite text fields (subject, body).
- **Step 3**: Click the `Classify Email` button to make the model predict if the input email is either spam or not spam.
  
  ![My Image](https://github.com/b-walls/UEBA-Spam-Email-Classification/blob/main/UI_Example.png)

## How to Use

- **Important Note**: This project has models that are run on my CUDA capable computer. You must have a CUDA capable machine in order to run these files the way it was intended, if you do not have a CUDA capable computer there are places in the code that are documented where you can change the device the models are run on. 

1. **Data Preparation**: Run the `data_preparation.ipynb` notebook to clean and prepare the dataset `enron_spam_data.csv`.
2. **Model Training**: Run the `model_training.ipynb` to train models (XGBClassifier, Random Forest Classifier).
3. **Model Classification**: Run the `run_classification.py`, see detailed steps in the `Model in Action` section above.

## Acknowledgements

- **Dataset**: The dataset used to train the spam classification models is derived from a GitHub repository, which compiled data from the study ["Spam Filtering with Naive Bayes - Which Naive Bayes?"](https://nes.aueb.gr/ipl/nlp/pubs/ceas2006_paper.pdf). You can access the dataset here: [Enron Spam Data](https://github.com/MWiechmann/enron_spam_data).
- **Sentiment Analysis Model**: [Hugging Face - CardiffNLP](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
