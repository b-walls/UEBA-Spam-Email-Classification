import re
import os
import nltk
import tkinter as tk
import pandas as pd
from tkinter import ttk, messagebox, filedialog
from pickle import load as pkl_load
from joblib import load as job_load
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer



class EmailClassifierApp:

    def __init__(self):
        # init model
        self.model = None
        self.data = None

        self.root = tk.Tk()
        self.root.title("Email Classification")
        self.root.geometry("800x500")

        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create menubar 
        self.menubar = tk.Menu(self.root)

        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label='Import', command=self._import)
        self.filemenu.add_separator()
        self.filemenu.add_command(label='Exit', command=self._close)

        self.menubar.add_cascade(menu=self.filemenu, label='File')
        self.root.config(menu=self.menubar)

        # Create a frame for the subject line
        self.subject_frame = ttk.Frame(self.main_frame)
        self.subject_frame.pack(fill=tk.X, pady=(0, 10))

        # Add a label and text box for the subject line
        self.subject_label = ttk.Label(self.subject_frame, text="Subject:")
        self.subject_label.pack(side=tk.LEFT, padx=(0, 10))
        self.subject_text = tk.Entry(self.subject_frame, width=70)
        self.subject_text.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Create a frame for the email body
        self.body_frame = ttk.Frame(self.main_frame)
        self.body_frame.pack(fill=tk.BOTH, expand=True)

        # Add a label and text box for the email body
        self.body_label = ttk.Label(self.body_frame, text="Body:")
        self.body_label.pack(anchor=tk.NW, padx=(0, 10))
        self.body_text = tk.Text(self.body_frame, height=20, width=70)
        self.body_text.pack(fill=tk.BOTH, expand=True)

        # Create a button frame
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=(10, 0))

        # Add a button at the bottom
        self.classify_button = ttk.Button(self.button_frame, text="Classify Email", command=self._classify_email)
        self.classify_button.pack(side=tk.RIGHT)

        self.root.mainloop()

    # method that classifies the input text with the imported model
    def _classify_email(self):
        subject = self.subject_text.get().strip()
        body = self.body_text.get('1.0', tk.END).strip()

        # error message for empty subject or body
        if subject == '' or body == '':
            messagebox.showinfo("Error", "Empty subject or body, please enter text into the appropriate fields.")
            return

        # error message for lack of model imported
        if self.model is None:
            messagebox.showinfo("Error", "No model selected. Please use the file menu to import a model.")
            return
        # error message for lack of imported data file
        elif self.data is None:
            messagebox.showinfo("Error", "No data file selected. Please use the file menu to import a data file.")
            return
        
        df = self._load_features(subject=subject, body=body)
        
        # fixes the dataframe to match the model's input 
        missing_cols = set(self.data.drop(columns={'Class'}).columns) - set(df.columns)
        missing_cols = list(missing_cols)
        new_columns = pd.DataFrame(0, index=df.index, columns=missing_cols)
        df = pd.concat([df, new_columns], axis=1)
        df = df[self.data.drop(columns={'Class'}).columns]
        prediction = self.model.predict(df)

        if prediction == 0:
            messagebox.showinfo("Classification", "Model classified as not spam.")
        else:
            messagebox.showinfo("Classification", "Model classified as spam.")
    
    # closes the window
    def _close(self):
        if messagebox.askyesno(title='', message='Are you sure?'):
            self.root.destroy()

    # opens dialog box file explorer to open data file and model to evaluate emails
    def _import(self):

        # opens file explorer to import model
        model_filename = filedialog.askopenfilename(
            initialdir=os.path.join(os.path.dirname(__file__), '.'),
            title='Select a Model',
            filetypes=[("Pickle Files (.pkl)", "*.pkl")]
        )

        
        # opens file explorer to import data file
        data_filename = filedialog.askopenfilename(
            initialdir=os.path.join(os.path.dirname(__file__), '.'),
            title='Select Training Data',
            filetypes=[("Pickle Files (.pkl)", "*.pkl")]
        )
        # checks if filepaths are empty 
        if model_filename == '' or data_filename == '':
            return
        
        with open(data_filename, 'rb') as file:
            self.data = pkl_load(file)

        self.model = job_load(model_filename)
    
    # loads the engineered features for the input email
    def _load_features(self, subject, body):
        df = pd.DataFrame({'Message': [body], 'Subject': subject})
        df['Message'] = df['Message'].astype(str)
        df['Subject'] = df['Subject'].astype(str)   

        # get engineered features
        
        # url counts
        df['urls_count_message'] = df['Message'].apply(self._count_url)
        df['urls_count_subject'] = df['Subject'].apply(self._count_url)
        df['urls_count'] = df['urls_count_message'] + df['urls_count_subject']
        df = df.drop(columns={'urls_count_message', 'urls_count_subject'})

        # special char counts
        df['special_chars_count_message'] = df['Message'].apply(self._count_special_chars)
        df['special_chars_count_subject'] = df['Subject'].apply(self._count_special_chars)
        df['special_chars_count'] = df['special_chars_count_message'] + df['special_chars_count_subject']
        df = df.drop(columns={'special_chars_count_message', 'special_chars_count_subject'})

        # urgent phrase counts
        df['urgent_phrase_count_message'] = df['Message'].apply(self._count_urgency_words)
        df['urgent_phrase_count_subject'] = df['Subject'].apply(self._count_urgency_words)
        df['urgent_phrase_count'] = df['urgent_phrase_count_message'] + df['urgent_phrase_count_subject']
        df = df.drop(columns={'urgent_phrase_count_message', 'urgent_phrase_count_subject'})

        # forwarded
        df['forwarded'] = df['Message'].apply(self._is_forwarded)

        df['Message'] = df['Message'].apply(self._clean_text)
        df['Subject'] = df['Subject'].apply(self._clean_text)

        # remove stopwords
        stop_words = stopwords.words()
        df['Message'] = df['Message'].apply(self._remove_stop_words, stop_words=stop_words)
        df['Subject'] = df['Subject'].apply(self._remove_stop_words, stop_words=stop_words)

        # lemmatize
        lemmatizer = WordNetLemmatizer()
        df['Message'] = df['Message'].apply(self._lemmatize_email, lemmatizer=lemmatizer)
        df['Subject'] = df['Subject'].apply(self._lemmatize_email, lemmatizer=lemmatizer)

        MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        model.to('cuda')

        
        # get polarity scores using pretrained LLM 
        msg = df.iloc[0]['Message']
        subject = df.iloc[0]['Subject']
        message_scores = self._polarity_scores(msg, model, tokenizer)
        subject_scores = self._polarity_scores(subject, model, tokenizer)
        message_results = {
            'msg_neg' : message_scores[0],
            'msg_neu' : message_scores[1],
            'msg_pos' : message_scores[2]
        }
        subject_results = {
            'sub_neg' : subject_scores[0],
            'sub_neu' : subject_scores[1],
            'sub_pos' : subject_scores[2]
        }
        results = message_results | subject_results

        results_df = pd.DataFrame([results])

        # concat the results to the data frame
        df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

        # run TF-IDF on the emails to vectorize the emails
        tfidf_msg = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
        X_msg = tfidf_msg.fit_transform(df['Message'])
        X_msg = pd.DataFrame(X_msg.toarray(), columns=('msg_freq_' + word for word in tfidf_msg.get_feature_names_out()))

        tfidf_sub = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        X_sub = tfidf_sub.fit_transform(df['Subject'])
        X_sub = pd.DataFrame(X_sub.toarray(), columns=('sub_freq_' + word for word in tfidf_sub.get_feature_names_out()))

        vectorized_df = pd.concat([X_msg, X_sub], axis=1)
        columns_to_add = df.drop(columns={'Message', 'Subject'}).columns
        vectorized_df[columns_to_add] = df[columns_to_add]

        # return the finished input data frame
        return vectorized_df

    # counts urls and hyperlinks
    @staticmethod
    def _count_url(text):
        # count the occurrences of 'http', 'https', and 'www'
        count_http = len(re.findall(r'http', text))
        count_https = len(re.findall(r'https', text))
        count_www = len(re.findall(r'www', text))
    
        # return the total count of urls in a new column
        return count_http + count_https + count_www
    
    # counts special characters
    @staticmethod
    def _count_special_chars(text):
        return len(re.findall(r'[!$%&]', text))

    # counts the occurances of urgent words and phrases
    @staticmethod
    def _count_urgency_words(text):
        urgency_words = [
            "immediate", "urgent", "critical", "important", "now", "ASAP", "as soon as possible",
            "emergency", "priority", "alert", "rush", "prompt", "hasten", "swift", "instantly",
            "right away", "without delay", "high priority", "imminent", "pressing", "time - sensitive",
            "expedite", "top priority", "crucial", "vital", "necessary", "quick", "speedy", "at once",
            "rapid", "flash", "instantaneous", "accelerated", "breakneck", "hurry", "immediately",
            "fast-track", "at the earliest", "act now", "don't delay", "on the double", "without hesitation",
            "fast", "soon", "now or never", "urgent action", "right now", "straightaway", "double-time",
            "speed", "express", "high-priority", "pressing need", "at your earliest convenience", "this instant",
            "forthwith", "like a shot", "snap to it", "on the spot", "no time to lose", "no delay",
            "in a hurry", "right this minute", "get going", "with haste"
        ]
        words = re.findall(r'\b\w+\b', text.lower())
        count = sum(1 for word in words if word in urgency_words)
        return count
    
    # returns length of text
    @staticmethod
    def _get_length(text):
        return len(text)
    
    # checks if the email is forwarded 
    @staticmethod
    def _is_forwarded(text):
        if (len(re.findall(r'-', text))) > 9 and len(re.findall(r'forward', text)) > 0:
            return 1
        else:
            return 0
    
    # removes special characters
    @staticmethod
    def _clean_text(text):
        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Remove excessive spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # removes stop words from the email
    @staticmethod
    def _remove_stop_words(text, stop_words):
        word_tokens = word_tokenize(text)
        new_text = [w for w in word_tokens if not w.lower() in stop_words]

        return ' '.join(new_text)
    
    # used for lemmatization
    @staticmethod
    def _get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    # lemmatize email
    @staticmethod
    def _lemmatize_email(text, lemmatizer):
        words = nltk.word_tokenize(text)
        lemmatized_words = [lemmatizer.lemmatize(w, EmailClassifierApp._get_wordnet_pos(w)) for w in words]
        return ' '.join(lemmatized_words)

    # runs the neural network 
    @staticmethod
    def _polarity_scores(text, model, tokenizer):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        scores = logits[0].cpu().detach().numpy()
        scores = softmax(scores)
        return scores
        
if __name__ == "__main__":
    app = EmailClassifierApp()
