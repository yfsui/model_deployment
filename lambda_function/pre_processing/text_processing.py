"""
Clean and tokenize text

Reference from Prof. Pierre-Hadrien Arnoux
"""

import re
from pre_processing.nltk_tokenize import TweetTokenizer

class TextProcessor():
    """
    change a string of raw text into an array of int
    """

    def __init__(self):
        self.tknzr = TweetTokenizer()

    def clean_text(self,string):
        """
        remove urls and unnecessary tokens
        """
        #remove link
        pattern1 = re.compile(r'https?://[A-Za-z0-9.,\/\'-:_\"@!&#…\n]+')
        text_without_link = pattern1.sub('', string)
        text_without_link = text_without_link.replace('\n', ' ')
        #remove hashtag
        pattern2 = re.compile(r'RT @[\w_]+: ')
        cleaned_text = pattern2.sub('', text_without_link)
        #remove punctuation marks
        pattern3 = re.compile(r'[^A-Za-z0-9\']')
        final_text = pattern3.sub(' ', cleaned_text)
        return final_text

    def tokenize_text(self,string):
        """
        change text into tokens
        """
        tokens = self.tknzr.tokenize(string)
        return tokens