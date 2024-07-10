Dataset: 
  The dataset used to train my spam classification models comes from this github post which compiled the data from a study
"Spam Filtering with Naive Bayes - Which Naive Bayes?": https://nes.aueb.gr/ipl/nlp/pubs/ceas2006_paper.pdf. 

The cleaner data comes from this repository: https://github.com/MWiechmann/enron_spam_data

Files:

data_preparation.ipynb -

  File used for data preparation for use in spam classificaiton models. This file contains a portion of feature engineering, and all of the text cleaning (removing special characters, removing stop-words, lemmitization). Due to the time complexity of the tasks and the amount of semi-raw data this file is separate from the rest of the project. This also allows for easier data management and storage, making it ready for the model when training and testing. 

  Used a tuned pretrained model for sentiment analysis on the emails as an extra insight into the emails. The tuned model comes from this huggingface post https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest.
