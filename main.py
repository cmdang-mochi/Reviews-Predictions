# This Python script makes predictions for reviews using 4 different categories.:

# `L+` : good/positive localization
# `L-` : bad/negative localization
# `RL` : localization request
# `YL` : localization exists

# Please set up the proper environment using the requirements.txt
# Please also download and unzip the pre-trained model from this github repo release 
# (Due to size limitations, the pre-trained model is located in the release)
# Update pathfile for the model load code in line 332 to where you have it on your directory after unzipping the file

# To run please use the terminal and type the following:    python main.py

import tkinter as tk
from tkinter import ttk
from time import sleep
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
from tkinter.messagebox import showerror

import pandas as pd
import numpy as np
import re
import PySimpleGUI as sg
import time

import nltk

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import tensorflow as tf
from tensorflow import keras

import tensorflow_hub as hub
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

from bert import bert_tokenization

CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}


# --- classes ---

class App(object):
    def __init__(self, parent):

        self.parent = parent

        self.filename = None
        self.df = None

        self.text = tk.Text(self.parent)
        self.text.pack()

        # load data button
        self.load_button = tk.Button(self.parent, text='Load Data', command=self.load)
        self.load_button.pack()

        # run script button
        self.script_button = tk.Button(self.parent, text='Get Predictions', command=self.script_python)
        self.script_button.pack()

        # save data button
        self.save_button = tk.Button(self.parent, text='Save Predictions', command=self.file_save)
        self.save_button.pack()

    def load(self):
        name = askopenfilename(filetypes=[('CSV', '*.csv',), ('Excel', ('*.xls', '*.xslm', '*.xlsx', 'xlrd'))])

        if name:
            if name.endswith('.csv'):
                self.df = pd.read_csv(name)
            else:
                self.df = pd.read_excel(name)

            self.filename = name

            # display directly
            self.text.insert('end', str(self.df.head()) + '\n')

        return self.df


    def script_python(self):

        # layout the window
        layout = [[sg.Text('This might take a while, please wait while predictions are being made.\n'
                           '\nThis window will close automatically when predictions are completed.\n'
                           '\nPlease save the file after it completes.')],
                  [sg.ProgressBar(1000, orientation='h', size=(20, 20), key='progress')]]

        # This Creates the Physical Window
        window = sg.Window('Processing data', layout).Finalize()
        progress_bar = window.FindElement('progress')

        # This Updates the Window
        # progress_bar.UpdateBar(Current Value to show, Maximum Value to show)
        progress_bar.UpdateBar(0, 5)
        # adding time.sleep(length in Seconds) has been used to Simulate adding your script in between Bar Updates
        time.sleep(.5)


        # Create empty dataframe
        df_pred = pd.DataFrame()

        # lowercase column headings just incase file input has capitals
        self.df.columns = [columns.lower() for columns in self.df.columns]

        # Function to clean text
        def clear_text(text):
            clean_text = re.sub(r'[^a-zA-z\']', ' ', text)
            clean_text = ' '.join(clean_text.split())
            clean_text = clean_text.lower()
            return (clean_text)

        # Function to expand contractions in text
        def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
            contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                              flags=re.IGNORECASE | re.DOTALL)
            def expand_match(contraction):
                match = contraction.group(0)
                first_char = match[0]
                expanded_contraction = contraction_mapping.get(match) \
                    if contraction_mapping.get(match) \
                    else contraction_mapping.get(match.lower())
                expanded_contraction = first_char + expanded_contraction[1:]
                return expanded_contraction
            expanded_text = contractions_pattern.sub(expand_match, text)
            expanded_text = re.sub("'", "", expanded_text)
            return expanded_text

        # Maps POS tag for any word types and returns noun
        def get_wordnet_pos(word):
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ,
                        "N": wordnet.NOUN,
                        "V": wordnet.VERB,
                        "R": wordnet.ADV}
            return tag_dict.get(tag, wordnet.NOUN)

        wordnet_lemmatizer = WordNetLemmatizer()

        # Used to convert words to noun based
        def lemmatize_tokens_nltk(text):
            lemmatized_text = []
            for i in nltk.word_tokenize(text):
                lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(i, get_wordnet_pos(i))))
            return ' '.join(lemmatized_text)

        # Preprocess review text/ Normalize text
        df_temp = self.df.copy()

        progress_bar.UpdateBar(1, 5)
        time.sleep(.5)

        df_temp['review_norm'] = df_temp['review'].apply(clear_text)
        df_temp['review_norm'] = df_temp['review_norm'].apply(expand_contractions)

        progress_bar.UpdateBar(2, 5)
        time.sleep(.5)

        df_temp['lemm_text'] = df_temp['review_norm'].apply(lambda x: lemmatize_tokens_nltk(x))

        progress_bar.UpdateBar(3, 5)
        time.sleep(.5)

        # BERT word embeddings matrix
        module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
        bert_layer = hub.KerasLayer(module_url, trainable=True)

        # create a tokenizer
        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

        # create token-embedding mapping
        def bert_encode(texts, tokenizer, max_len=512):
            all_tokens = []
            all_masks = []
            all_segments = []
            for text in texts:
                # tokenize text
                text = tokenizer.tokenize(text)
                # convert text to sequence of tokens and pad them to ensure equal length vectors
                text = text[:max_len - 2]
                input_sequence = ["[CLS]"] + text + ["[SEP]"]
                pad_len = max_len - len(input_sequence)
                tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
                pad_masks = [1] * len(input_sequence) + [0] * pad_len
                segment_ids = [0] * max_len
                all_tokens.append(tokens)
                all_masks.append(pad_masks)
                all_segments.append(segment_ids)
            return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

        # Vectorize text or add Word Embeddings
        max_len = 150
        text_vector = bert_encode(df_temp['lemm_text'], tokenizer, max_len=max_len)

        # load the best model
        best_model = keras.models.load_model('keras_bert_model_unzip/keras_bert_model', compile=True) #Please udpate path file for where the pre-trained model is located

        # compute the predicted probabilities for text for each mark
        y_prob = best_model.predict(text_vector, verbose=1)

        progress_bar.UpdateBar(4, 5)
        time.sleep(.5)

        # write columns for each predicted probabilities mark
        self.df['prob_L+'] = y_prob[:, 0]
        self.df['prob_L-'] = y_prob[:, 1]
        self.df['prob_RL'] = y_prob[:, 2]
        self.df['prob_YL'] = y_prob[:, 3]

        # write data to empty dataframe
        df_pred = df_pred.append(self.df)

        progress_bar.UpdateBar(5, 5)
        time.sleep(.5)
        # I paused for 3 seconds at the end to give you time to see it has completed before closing the window
        time.sleep(3)

        # done with loop... need to destroy the window as it's still open
        window.close()

        popup = tk.Toplevel()
        tk.Label(popup, text="Please save the predictions.").grid(row=0, column=0)

        return self.df

    def file_save(self):
        files = [('Excel files', '*.xlsx'), ('All Files', '*.*')]
        fname = asksaveasfilename(filetypes=files, defaultextension=files)

        self.df.to_excel('{fname}{ext}'.format(fname=fname, ext='.xlsx'), index=False)


# --- main ---

if __name__ == '__main__':
    root = tk.Tk()
    root.title('Making Reviews Predictions')
    root.config(bg='#345')
    top = App(root)
    root.mainloop()
