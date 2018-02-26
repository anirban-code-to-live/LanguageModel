import math
from nltk.probability import SimpleGoodTuringProbDist
from nltk.probability import FreqDist
from nltk.util import ngrams

# used for unseen words in training vocabularies
UNK = None
# sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"


class UnigramModel:
    def __init__(self, sentences, smoothing="AddOne"):
        self.unigram_frequencies = dict()
        self.corpus_length = 0
        for sentence in sentences:
            unigrams = ngrams(sentence, 1, pad_left=False, pad_right=False)
            for unigram in unigrams:
                self.unigram_frequencies[unigram] = self.unigram_frequencies.get(unigram, 0) + 1
                if unigram != SENTENCE_START and unigram != SENTENCE_END:
                    self.corpus_length += 1
        # subtract 2 because unigram_frequencies dictionary contains values for SENTENCE_START and SENTENCE_END
        self.unique_words = len(self.unigram_frequencies) - 2
        self.smoothing = smoothing
        self._unigram_good_turing = SimpleGoodTuringProbDist(freqdist=FreqDist(self.unigram_frequencies))

    def calculate_unigram_probability(self, word):
        if self.smoothing == "GoodTuring":
            return self._unigram_good_turing.prob(word)
        elif self.smoothing == "AddOne":
            word_probability_numerator = self.unigram_frequencies.get(word, 0) + 1
            # add one more to total number of seen unique words for UNK - unseen events
            word_probability_denominator = self.corpus_length + self.unique_words + 1
            return float(word_probability_numerator) / float(word_probability_denominator)
        else:
            try:
                raise ValueError("Supported smoothing techniques - 1. AddOne 2. GoodTuring")
            except ValueError as error:
                print(error.args)

    def calculate_sentence_probability(self, sentence):
        sentence_probability_log_sum = 0
        for word in sentence:
            if word != SENTENCE_START and word != SENTENCE_END:
                word_probability = self.calculate_unigram_probability(word)
                sentence_probability_log_sum += math.log(word_probability, 2)
        return sentence_probability_log_sum

    def sorted_vocabulary(self):
        full_vocab = list(self.unigram_frequencies.keys())
        full_vocab.sort()
        full_vocab.append(UNK)
        full_vocab.append(SENTENCE_START)
        full_vocab.append(SENTENCE_END)
        return full_vocab

    def calculate_number_of_unigrams(self, sentences):
        unigram_count = 0
        for sentence in sentences:
            unigram_count += len(sentence)
        return unigram_count

    def calculate_unigram_perplexity(self, sentences):
        unigram_count = self.calculate_number_of_unigrams(sentences)
        sentence_probability_log_sum = 0
        for sentence in sentences:
            try:
                sentence_probability_log_sum -= self.calculate_sentence_probability(sentence)
            except (RuntimeError, ValueError):
                sentence_probability_log_sum -= float('-inf')
        return math.pow(2, sentence_probability_log_sum / unigram_count)