import math
import random
from nltk.util import ngrams

# sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"


class MixgramModel:
    def __init__(self, lambdas, models):
        self._unigram_model = models[0]
        self._bigram_model = models[1]
        self._trigram_model = models[2]
        self._lambdas = lambdas
        self.highest_order = 3

    def calculate_mixgram_probabilty(self, previous_to_previous_word, previous_word, word):
        trigram_probability = self._trigram_model.cal_prob_good_turing_trigram(previous_to_previous_word, previous_word, word)
        bigram_probability = self._bigram_model.cal_prob_good_turing_bigram(previous_word, word)
        unigram_probability = self._unigram_model.cal_prob_good_turing(word)
        mixgram_probability = self._lambdas[0] * unigram_probability + self._lambdas[1] * bigram_probability \
                              + self._lambdas[2] * trigram_probability
        # print(mixgram_probability)
        return mixgram_probability

    def calculate_mixgram_sentence_probability(self, sentence, normalize_probability=False):
        mixgram_sentence_probability_log_sum = 0
        trigrams = ngrams(sentence, 3, pad_left=True, pad_right=True, left_pad_symbol=SENTENCE_START,
                          right_pad_symbol=SENTENCE_END)
        for trigram in trigrams:
                mixgram_word_probability = self.calculate_mixgram_probabilty(trigram[0], trigram[1], trigram[2])
                # if mixgram_word_probability != 0:
                mixgram_sentence_probability_log_sum += math.log(mixgram_word_probability, 2)
        return math.pow(2, mixgram_sentence_probability_log_sum) if normalize_probability \
            else mixgram_sentence_probability_log_sum

    def calculate_number_of_mixgrams(self, sentences):
        mixgram_count = 0
        for sentence in sentences:
            mixgram_count += len(sentence) + 2
        return mixgram_count

    def calculate_mixgram_perplexity(self, sentences):
        number_of_mixgrams = self.calculate_number_of_mixgrams(sentences)
        mixgram_sentence_probability_log_sum = 0
        for sentence in sentences:
            try:
                mixgram_sentence_probability_log_sum -= self.calculate_mixgram_sentence_probability(sentence)
            except RuntimeError:
                mixgram_sentence_probability_log_sum -= float('-inf')
        return math.pow(2, mixgram_sentence_probability_log_sum / number_of_mixgrams)

    def generate_sentence(self, min_length=8):
        frequencies = self._trigram_model.get_trigram_frequency_list()
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