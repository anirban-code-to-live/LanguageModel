from src import bigram
import math
from nltk.util import ngrams
from nltk.probability import FreqDist
from nltk.probability import SimpleGoodTuringProbDist
import random

# sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"


class TrigramModel(bigram.BigramModel):
    def __init__(self, sentences, words, smoothing="AddOne"):
        bigram.BigramModel.__init__(self, sentences, words, smoothing)
        self.highest_order = 3
        self.trigram_frequencies = dict()
        self.unique_trigrams = set()
        for sentence in sentences:
            trigrams = ngrams(sentence, 3, pad_left=True, pad_right=True, left_pad_symbol=SENTENCE_START, right_pad_symbol= SENTENCE_END)
            for trigram in trigrams:
                self.trigram_frequencies[trigram] = self.trigram_frequencies.get(trigram, 0) + 1
                self.unique_trigrams.add(trigram)
        self.unique__trigram_words = len(self.unique_trigrams)
        self._trigram_good_turing_estimator = SimpleGoodTuringProbDist(FreqDist(self.trigram_frequencies))

    def calculate_trigram_probabilty(self, previous_to_previous_word, previous_word, word):
        if self.smoothing == "GoodTuring":
            return self._trigram_good_turing_estimator.prob((previous_to_previous_word, previous_word, word))
        elif self.smoothing == "AddOne":
            trigram_word_probability_numerator = self.trigram_frequencies.get(
                (previous_to_previous_word, previous_word, word), 0) + 1
            trigram_word_probability_denominator = self.bigram_frequencies.get(
                (previous_to_previous_word, previous_word), 0) + self.unique__trigram_words
            return float(trigram_word_probability_numerator) / float(trigram_word_probability_denominator)
        else:
            try:
                raise ValueError("Supported smoothing techniques - 1. AddOne 2. GoodTuring")
            except ValueError as error:
                print(error.args)

    def calculate_trigram_sentence_probability(self, sentence):
        trigram_sentence_probability_log_sum = 0
        trigrams = ngrams(sentence, 3, pad_left=True, pad_right=True, left_pad_symbol=SENTENCE_START,
                          right_pad_symbol=SENTENCE_END)
        for trigram in trigrams:
                trigram_word_probability = self.calculate_trigram_probabilty(trigram[0], trigram[1], trigram[2])
                trigram_sentence_probability_log_sum += math.log(trigram_word_probability, 2)
        return trigram_sentence_probability_log_sum

    def calculate_number_of_trigrams(self, sentences):
        trigram_count = 0
        for sentence in sentences:
            trigram_count += len(sentence) + 2
        return trigram_count

    def calculate_trigram_perplexity(self, sentences):
        number_of_trigrams = self.calculate_number_of_trigrams(sentences)
        trigram_sentence_probability_log_sum = 0
        for sentence in sentences:
            try:
                trigram_sentence_probability_log_sum -= self.calculate_trigram_sentence_probability(sentence)
            except RuntimeError:
                trigram_sentence_probability_log_sum -= float('-inf')
        return math.pow(2, trigram_sentence_probability_log_sum / number_of_trigrams)

    def get_trigram_frequency_list(self):
        return self.trigram_frequencies

    def generate_sentence(self, min_length=8):
        frequencies = self.get_trigram_frequency_list()
        sent = [SENTENCE_START] * (self.highest_order - 1)
        while len(sent) < min_length + self.highest_order:
            word = self._generate_next_word(sent, frequencies)
            sent.append(word)
        sent = ' '.join(sent[(self.highest_order - 1):(self.highest_order - 1 + min_length)])
        return sent

    def _generate_next_word(self, sent, frequencies):
        context = tuple(self._get_context(sent))
        ngrams_with_freq = list(
            (ngram, freq) for ngram, freq in frequencies.items()
            if ngram[:-1] == context)
        if len(ngrams_with_freq) == 0:
            rand_int = random.randint(0, len(frequencies))
            return list(frequencies.keys())[rand_int][1]
        _, max_freq = max(ngrams_with_freq, key=lambda x: x[1])
        ngrams_with_freq = list((ngram, math.exp(freq - max_freq)) for ngram, freq in ngrams_with_freq)
        total_freq = sum(freq for ngram, freq in ngrams_with_freq)
        ngrams_with_prob = list((ngram, freq / total_freq) for ngram, freq in ngrams_with_freq)
        rand = random.random()
        for ngram, prob in ngrams_with_prob:
            rand -= prob
            if rand < 0:
                return ngram[-1]
        return ngrams_with_prob[0][0][-1]

    def _get_context(self, sentence):
        return sentence[(len(sentence) - self.highest_order + 1):]