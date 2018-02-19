from src import unigram
import math
import numpy as np
from src import tokenization

# used for unseen words in training vocabularies
UNK = None
# sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"


class BigramModel(unigram.UnigramModel):
    def __init__(self, sentences, words, smoothing=False):
        unigram.UnigramModel.__init__(self, words, smoothing)
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
        # we subtracted two for the Unigram model as the unigram_frequencies dictionary
        # contains values for SENTENCE_START and SENTENCE_END but these need to be included in Bigram
        self.unique__bigram_words = len(self.unigram_frequencies)

    def calculate_bigram_probabilty(self, previous_word, word):
        bigram_word_probability_numerator = self.bigram_frequencies.get((previous_word, word), 0)
        bigram_word_probability_denominator = self.unigram_frequencies.get(previous_word, 0)
        if self.smoothing:
            bigram_word_probability_numerator += 1
            bigram_word_probability_denominator += self.unique__bigram_words
        return 0.0 if bigram_word_probability_numerator == 0 or bigram_word_probability_denominator == 0 else float(
            bigram_word_probability_numerator) / float(bigram_word_probability_denominator)

    def calculate_bigram_sentence_probability(self, sentence, normalize_probability=True):
        bigram_sentence_probability_log_sum = 0
        previous_word = SENTENCE_START
        print(tokenization.tokenize_words(sentence))
        for word in tokenization.tokenize_words(sentence):
            if previous_word is not None:
                bigram_word_probability = self.calculate_bigram_probabilty(previous_word, word)
                if bigram_word_probability != 0:
                    bigram_sentence_probability_log_sum += math.log(bigram_word_probability, 2)
            previous_word = word
        return 0.0 if bigram_sentence_probability_log_sum == 0 else math.pow(2, bigram_sentence_probability_log_sum) \
            if normalize_probability else bigram_sentence_probability_log_sum

    def calculate_number_of_bigrams(self, sentences):
        bigram_count = 0
        for sentence in tokenization.tokenize_sentence(sentences):
            bigram_count += len(tokenization.tokenize_words(sentence)) + 1
        return bigram_count

    def calculate_bigram_perplexity(self, sentences):
        number_of_bigrams = self.calculate_number_of_bigrams(sentences)
        bigram_sentence_probability_log_sum = 0
        for sentence in tokenization.tokenize_sentence(sentences):
            try:
                bigram_sentence_probability_log_sum -= math.log(self.calculate_bigram_sentence_probability(sentence), 2)
            except RuntimeError:
                bigram_sentence_probability_log_sum -= float('-inf')
        return math.pow(2, bigram_sentence_probability_log_sum / number_of_bigrams)