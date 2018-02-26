from src import unigram
import math
import numpy as np
from nltk.probability import FreqDist
from nltk.probability import SimpleGoodTuringProbDist

# sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"


class BigramModel(unigram.UnigramModel):
    def __init__(self, sentences, smoothing="AddOne"):
        unigram.UnigramModel.__init__(self, sentences, smoothing)
        self.bigram_frequencies = dict()
        self.unique_bigrams = set()
        for sentence in sentences:
            words = np.insert(np.append(sentence, SENTENCE_END), 0, SENTENCE_START)
            previous_word = None
            for word in words:
                if previous_word is not None:
                    self.bigram_frequencies[(previous_word, word)] = self.bigram_frequencies.get((previous_word, word), 0) + 1
                    if previous_word != SENTENCE_START and word != SENTENCE_END:
                        self.unique_bigrams.add((previous_word, word))
                previous_word = word
        self.unique__bigram_words = len(self.unique_bigrams)
        self._bigram_good_turing = SimpleGoodTuringProbDist(freqdist=FreqDist(self.bigram_frequencies))

    def calculate_bigram_probabilty(self, previous_word, word):
        if self.smoothing == "GoodTuring":
            return self._bigram_good_turing.prob((previous_word, word))
        elif self.smoothing == "AddOne":
            bigram_word_probability_numerator = self.bigram_frequencies.get((previous_word, word), 0) + 1
            bigram_word_probability_denominator = self.unigram_frequencies.get(previous_word, 0) + self.unique__bigram_words
            return float(bigram_word_probability_numerator) / float(bigram_word_probability_denominator)
        else:
            try:
                raise ValueError("Supported smoothing techniques - 1. AddOne 2. GoodTuring")
            except ValueError as error:
                print(error.args)

    def calculate_bigram_sentence_probability(self, sentence):
        bigram_sentence_probability_log_sum = 0
        previous_word = SENTENCE_START
        for word in sentence:
            if previous_word is not None:
                bigram_word_probability = self.calculate_bigram_probabilty(previous_word, word)
                bigram_sentence_probability_log_sum += math.log(bigram_word_probability, 2)
            previous_word = word
        return bigram_sentence_probability_log_sum

    def calculate_number_of_bigrams(self, sentences):
        bigram_count = 0
        for sentence in sentences:
            bigram_count += len(sentence) + 1
        return bigram_count

    def calculate_bigram_perplexity(self, sentences):
        number_of_bigrams = self.calculate_number_of_bigrams(sentences)
        bigram_sentence_probability_log_sum = 0
        for sentence in sentences:
            try:
                bigram_sentence_probability_log_sum -= self.calculate_bigram_sentence_probability(sentence)
            except RuntimeError:
                bigram_sentence_probability_log_sum -= float('-inf')
        return math.pow(2, bigram_sentence_probability_log_sum / number_of_bigrams)