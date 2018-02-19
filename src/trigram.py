from src import bigram
import math
from nltk.util import ngrams
from src import tokenization

# used for unseen words in training vocabularies
UNK = None
# sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"


class TrigramModel(bigram.BigramModel):
    def __init__(self, sentences, words, smoothing=False):
        bigram.BigramModel.__init__(self, sentences, words, smoothing)
        self.trigram_frequencies = dict()
        self.unique_trigrams = set()
        for sentence in sentences:
            # words = tokenization.tokenize_words(sentence)
            trigrams = ngrams(sentence, 3, pad_left=True, pad_right=True, left_pad_symbol=SENTENCE_START, right_pad_symbol= SENTENCE_END)
            for trigram in trigrams:
                self.trigram_frequencies[trigram] = self.trigram_frequencies.get(trigram, 0) + 1
                self.unique_trigrams.add(trigram)
        # we subtracted two for the Unigram model as the unigram_frequencies dictionary
        # contains values for SENTENCE_START and SENTENCE_END but these need to be included in Bigram
        self.unique__trigram_words = len(self.unigram_frequencies)

    def calculate_trigram_probabilty(self, previous_to_previous_word, previous_word, word):
        trigram_word_probability_numerator = self.trigram_frequencies.get((previous_to_previous_word, previous_word, word), 0)
        trigram_word_probability_denominator = self.bigram_frequencies.get((previous_to_previous_word, previous_word), 0)
        if self.smoothing:
            trigram_word_probability_numerator += 1
            trigram_word_probability_denominator += self.unique__trigram_words
        return 0.0 if trigram_word_probability_numerator == 0 or trigram_word_probability_denominator == 0 else float(
            trigram_word_probability_numerator) / float(trigram_word_probability_denominator)

    def calculate_trigram_sentence_probability(self, sentence, normalize_probability=True):
        trigram_sentence_probability_log_sum = 0
        print(tokenization.tokenize_words(sentence))
        words = tokenization.tokenize_words(sentence)
        trigrams = ngrams(words, 3, pad_left=True, pad_right=True, left_pad_symbol=SENTENCE_START,
                          right_pad_symbol=SENTENCE_END)
        for trigram in trigrams:
                trigram_word_probability = self.calculate_trigram_probabilty(trigram[0], trigram[1], trigram[2])
                if trigram_word_probability != 0:
                    trigram_sentence_probability_log_sum += math.log(trigram_word_probability, 2)
        return 0.0 if trigram_sentence_probability_log_sum == 0 else math.pow(2, trigram_sentence_probability_log_sum) \
            if normalize_probability else trigram_sentence_probability_log_sum

    def calculate_number_of_trigrams(self, sentences):
        trigram_count = 0
        for sentence in tokenization.tokenize_sentence(sentences):
            trigram_count += len(tokenization.tokenize_words(sentence)) + 2
        return trigram_count

    def calculate_trigram_perplexity(self, sentences):
        number_of_trigrams = self.calculate_number_of_trigrams(sentences)
        trigram_sentence_probability_log_sum = 0
        for sentence in tokenization.tokenize_sentence(sentences):
            try:
                trigram_sentence_probability_log_sum -= math.log(self.calculate_trigram_sentence_probability(sentence), 2)
            except RuntimeError:
                trigram_sentence_probability_log_sum -= float('-inf')
        return math.pow(2, trigram_sentence_probability_log_sum / number_of_trigrams)

    def get_trigram_frequency_list(self):
        return self.trigram_frequencies