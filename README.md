Spam Classification Model
Dataset
The dataset used to train the spam classification models is derived from a GitHub repository, which compiled data from the study "Spam Filtering with Naive Bayes - Which Naive Bayes?". You can access the dataset here: Enron Spam Data.

Data Preparation
data_preparation.ipynb
The data_preparation.ipynb notebook is dedicated to the preparation of data for training spam classification models. This notebook includes:

Feature Engineering: A portion of the feature engineering process is documented here.
Text Cleaning: Comprehensive text cleaning procedures, including removing special characters, stop-word removal, and lemmatization, are implemented.
Given the computational complexity and the volume of semi-raw data, the data preparation tasks are separated from the main project. This separation facilitates easier data management and storage, ensuring the data is ready for model training and testing.

Sentiment Analysis
Additionally, a pre-tuned sentiment analysis model was used to gain extra insights into the emails. This model is sourced from a Hugging Face post and provides sentiment scores for the email texts.

How to Use
Data Preparation: Run the data_preparation.ipynb notebook to clean and prepare the dataset.
Model Training: Use the prepared data for training your spam classification models.
Sentiment Analysis: Utilize the sentiment analysis insights for further understanding of the email content.
Acknowledgements
Dataset: Enron Spam Data
Sentiment Analysis Model: Hugging Face - CardiffNLP
References
"Spam Filtering with Naive Bayes - Which Naive Bayes?"
