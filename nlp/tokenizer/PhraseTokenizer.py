from typing import Union, Tuple, List, Iterable, Dict
import collections
import string
import os
import json
import logging
from .WordTokenizer import WordTokenizer, ENGLISH_STOP_WORDS
import nltk

logger = logging.getLogger(__name__)


class PhraseTokenizer(WordTokenizer):

    def __init__(self,
                 vocab: Iterable[str]=[], 
                 stop_words: Iterable[str] = ENGLISH_STOP_WORDS,
                 do_lower_cae : bool =False,
                 ngram_separator:str = '_', 
                 max_ngram_length: int = 5):
        self.stop_words = set(stop_words)
        self.do_lower_case = do_lower_cae
        self.ngram_separator = ngram_separator
        self.max_ngram_length = max_ngram_length
        self.set_vocab(vocab)

    def get_vocab(self):
        return self.vocab

    def set_vocab(self, vocab:Iterable[str]):
        self.vocab = vocab
        self.word2idx = collections.OrderDict([(word, idx) for idx, word in enumerate(vocab)])

        self.ngram_lookup = set()
        self.ngram_lengths = set()
        for word in vocab:

            if self.ngram_separator is not None and self.ngram_separator in word:
                ngram_count = word.count(self.ngram_separator) + 1
                if self.ngram_separator + self.ngram_separator not in word and ngram_count <= self.max_ngram_length:
                    self.ngram_lookup.add(word)
                    self.ngram_lengths.add(ngram_count)

                if len(vocab)> 0:
                    logger.info(f"PhraseTokenizer - Phrase ngram lengths: {self.ngram_lengths} ")
                    logger.info(f"PhraseTokenizer - num Phrases: {self.ngram_lookup} ")

    def tokenize(self, text: str) ->List[int]:
        tokens = nltk.word_tokenize(text, preserve_line=True)

        for ngram_len  in sorted(self.ngram_lengths, reverse=True):
            idx = 0
            while idx <= len(tokens) - ngram_len:
                ngram = self.ngram_separator.join(tokens[idx:idx + ngram_len])

                if ngram in self.ngram_lookup:
                    tokens[idx:idx+ngram_len] = [ngram]
                elif ngram.lower() in self.ngram_lookup:
                    tokens[idx:idx+ngram_len] = [ngram.lower()]
                idx += 1
        
        tokens_filtered = []
        for token



    
