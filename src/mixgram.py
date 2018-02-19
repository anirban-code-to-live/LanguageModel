import math
import random
from nltk.util import ngrams
from src import tokenization

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
        trigram_probability = self._trigram_model.calculate_trigram_probabilty(previous_to_previous_word, previous_word, word)
        bigram_probability = self._bigram_model.calculate_bigram_probabilty(previous_word, word)
        unigram_probability = self._unigram_model.calculate_unigram_probability(word)
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
                if mixgram_word_probability != 0:
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

    def _get_trigram_probs(self):
        return self._trigram_model.get_trigram_frequency_list()

    def generate_sentence(self, min_length = 8):
        sent = []
        probs = self._get_trigram_probs()
        # print(probs)
        while len(sent) < min_length + self.highest_order:
            sent = [SENTENCE_START] * (self.highest_order - 1)
            # Append first to avoid case where start & end symbal are same
            sent.append(self._generate_next_word(sent, probs))
            while sent[-1] != SENTENCE_END:
                sent.append(self._generate_next_word(sent, probs))
        sent = ' '.join(sent[(self.highest_order - 1):(self.highest_order - 1 + min_length)])
        return sent

    def _generate_next_word(self, sent, probs):
        context = tuple(self._get_context(sent))
        pos_ngrams = list(
            (ngram, logprob) for ngram, logprob in probs.items()
            if ngram[:-1] == context)
        # print(pos_ngrams)
        # Normalize to get conditional probability.
        # Subtract max logprob from all logprobs to avoid underflow.
        _, max_logprob = max(pos_ngrams, key=lambda x: x[1])
        # print(max_logprob)
        pos_ngrams = list((ngram, math.exp(prob - max_logprob)) for ngram, prob in pos_ngrams)
        total_prob = sum(prob for ngram, prob in pos_ngrams)
        pos_ngrams = list((ngram, prob / total_prob) for ngram, prob in pos_ngrams)
        rand = random.random()
        for ngram, prob in pos_ngrams:
            rand -= prob
            if rand < 0:
                return ngram[-1]
        return ngram[-1]

    def _get_context(self, sentence):
        return sentence[(len(sentence) - self.highest_order + 1):]