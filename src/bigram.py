from src import unigram
import math
import numpy as np
from nltk.probability import FreqDist
from nltk.probability import SimpleGoodTuringProbDist

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
        self.unique__bigram_words = len(self.unique_bigrams)
        freq_dist = FreqDist(self.bigram_frequencies)
        self._good_turing = SimpleGoodTuringProbDist(freqdist=freq_dist)

    def cal_prob_good_turing_bigram(self, previous_word, word):
        return self._good_turing.prob((previous_word, word))

    def calculate_bigram_probabilty(self, previous_word, word):
        bigram_word_probability_numerator = self.bigram_frequencies.get((previous_word, word), 0)
        bigram_word_probability_denominator = self.unigram_frequencies.get(previous_word, 0)

        # Katz smoothing
        # if bigram_word_probability_numerator != 0:
        #     bigram_word_probability_numerator = 0.9995 * bigram_word_probability_numerator
        # else:
        #     sum_prob = 0
        #     for current_word in self.sorted_vocabulary():
        #         count = self.bigram_frequencies.get((previous_word, current_word), 0)
        #         if count == 0:
        #             sum_prob += self.calculate_unigram_probability(current_word)
        #     bigram_word_probability_numerator = 0.0005 * bigram_word_probability_denominator * self.calculate_unigram_probability(word)/sum_prob
        if self.smoothing:
            bigram_word_probability_numerator += 1
            bigram_word_probability_denominator += self.unique__bigram_words
        return 0.0 if bigram_word_probability_numerator == 0 or bigram_word_probability_denominator == 0 else float(
            bigram_word_probability_numerator) / float(bigram_word_probability_denominator)

    def calculate_bigram_sentence_probability(self, sentence, normalize_probability=False):
        bigram_sentence_probability_log_sum = 0
        previous_word = SENTENCE_START
        for word in sentence:
            if previous_word is not None:
                bigram_word_probability = self.cal_prob_good_turing_bigram(previous_word, word)
                # if bigram_word_probability != 0:
                bigram_sentence_probability_log_sum += math.log(bigram_word_probability, 2)
            previous_word = word
        return 0.0 if bigram_sentence_probability_log_sum == 0 else math.pow(2, bigram_sentence_probability_log_sum) \
            if normalize_probability else bigram_sentence_probability_log_sum

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