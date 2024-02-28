import numpy as np
import pandas as pd
import nltk as nltk
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import pos_tag

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


def clean_text(text):
    punctuation_re = '[?؟!٪,،@#$%&*€+-£_~\“̯/=><.\۰):؛}{÷%("\'ًٌٍَُِّْ٠-٩]'
    emoji_re = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642"
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
            "]+", flags=re.UNICODE)

    stop_words = set(stopwords.words("arabic"))
    stemmer = SnowballStemmer('arabic')
    # remove punct
    no_punc = re.sub(punctuation_re, ' ', str(text))

    # # remove non-arabic letters and emojis
    no_english = re.sub(r'[a-zA-Z?]', ' ', no_punc)

    # remove emojis
    no_emojis = emoji_re.sub('', no_punc)

    text_no_numbers = re.sub(r'[0-9]',' ', no_emojis)

    text_no_stopwords = nltk.wordpunct_tokenize(text_no_numbers)
    words_no_stopwords = [word for word in text_no_stopwords if word not in stop_words]

    # join the words into a single string
    cleaned_text = ' '.join(words_no_stopwords)

    # tokenize
    tokens = nltk.word_tokenize(cleaned_text)

    # Stemming
    stemmedWords = [stemmer.stem(word) for word in tokens]

    return ' '.join(stemmedWords)
