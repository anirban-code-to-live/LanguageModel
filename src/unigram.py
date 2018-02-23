import math
from nltk.probability import SimpleGoodTuringProbDist
from nltk.probability import FreqDist

# used for unseen words in training vocabularies
UNK = None
# sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"


class UnigramModel:
    def __init__(self, words, smoothing=False):
        self.unigram_frequencies = dict()
        self.corpus_length = 0
        for word in words:
            # print(word)
            self.unigram_frequencies[word] = self.unigram_frequencies.get(word, 0) + 1
            if word != SENTENCE_START and word != SENTENCE_END:
                self.corpus_length += 1
        # subtract 2 because unigram_frequencies dictionary contains values for SENTENCE_START and SENTENCE_END
        self.unique_words = len(self.unigram_frequencies) - 2
        # print(self.unique_words)
        # print(self.corpus_length)
        self.smoothing = smoothing
        freq_dist = FreqDist(words)
        self.good_turing = SimpleGoodTuringProbDist(freqdist=freq_dist)


    def cal_prob_good_turing(self, word):
        return self.good_turing.prob(word)

    def calculate_unigram_probability(self, word):
            word_probability_numerator = self.unigram_frequencies.get(word, 0)
            # print(word_probability_numerator)
            word_probability_denominator = self.corpus_length
            if word_probability_numerator == 0 & self.smoothing is False:
                word_probability_numerator = 1
            if self.smoothing:
                word_probability_numerator += 0.001*1
                # add one more to total number of seen unique words for UNK - unseen events
                word_probability_denominator += 0.001*(self.unique_words + 1)
            return 0.0 if word_probability_numerator == 0 else float(word_probability_numerator) / float(word_probability_denominator)

    def calculate_sentence_probability(self, sentence, normalize_probability=False):
        sentence_probability_log_sum = 0
        for word in sentence:
            # print(word)
            if word != SENTENCE_START and word != SENTENCE_END:
                word_probability = self.cal_prob_good_turing(word)
                # if word_probability != 0:
                sentence_probability_log_sum += math.log(word_probability, 2)
        return 0.0 if sentence_probability_log_sum == 0 else math.pow(2, sentence_probability_log_sum) \
            if normalize_probability else sentence_probability_log_sum

    def sorted_vocabulary(self):
        full_vocab = list(self.unigram_frequencies.keys())
        # full_vocab.remove(SENTENCE_START)
        # full_vocab.remove(SENTENCE_END)
        full_vocab.sort()
        full_vocab.append(UNK)
        full_vocab.append(SENTENCE_START)
        full_vocab.append(SENTENCE_END)
        return full_vocab

    # calculate number of unigrams
    def calculate_number_of_unigrams(self, sentences):
        unigram_count = 0
        for sentence in sentences:
            unigram_count += len(sentence)
        return unigram_count

    # calculate perplexty
    def calculate_unigram_perplexity(self, sentences):
        unigram_count = self.calculate_number_of_unigrams(sentences)
        sentence_probability_log_sum = 0
        for sentence in sentences:
            try:
                sentence_probability_log_sum -= self.calculate_sentence_probability(sentence)
            except (RuntimeError, ValueError):
                sentence_probability_log_sum -= float('-inf')
        return math.pow(2, sentence_probability_log_sum / unigram_count)