from src.bigram import BigramModel
from src.unigram import UnigramModel
from src.trigram import TrigramModel
from nltk import ngrams
from src.mixgram import MixgramModel
import numpy as np
from nltk.corpus import gutenberg
from nltk.corpus import brown
import os


if __name__ == '__main__':

    if not os.path.exists("../output_data"):
        os.makedirs("../output_data")
    output_file = open('../output_data/perplexity.txt', 'a')

    validation_set_sizes = [50, 200, 500, 1000]

    # sentences = gutenberg.sents()
    # words = gutenberg.words()
    # output_file.write("Train Corpus :: Gutenberg, Validation Corpus:: Brown \n")

    sentences = brown.sents()
    words = brown.words()
    output_file.write("\n\n\n")
    output_file.write("Train Corpus :: Brown, Validation Corpus:: Gutenberg \n")

    for validation_set_size in validation_set_sizes:
        validation_sentences = gutenberg.sents()[0:validation_set_size]

        unigram_model = UnigramModel(words, smoothing="GoodTuring")
        unigram_perplexity = unigram_model.calculate_unigram_perplexity(validation_sentences)
        print("Unigram perplexity :: " + str(unigram_perplexity))
        output_file.write("Unigram Validation Size :: " + str(validation_set_size) +
                          " Perplexity :: " + str(unigram_perplexity))
        output_file.write("\n\n")

        bigram_model = BigramModel(sentences, words, smoothing="GoodTuring")
        bigram_perplexity = bigram_model.calculate_bigram_perplexity(validation_sentences)
        print("Bigram perplexity :: " + str(bigram_perplexity))
        output_file.write("Bigram Validation Size :: " + str(validation_set_size) +
                          " Perplexity :: " + str(bigram_perplexity))
        output_file.write("\n\n")

        trigram_model = TrigramModel(sentences, words, smoothing="GoodTuring")
        # prob = trigram_model.calculate_trigram_probabilty("hey", "there", "the")
        trigram_perplexity = trigram_model.calculate_trigram_perplexity(validation_sentences)
        print("Trigram perplexity :: " + str(trigram_perplexity))
        output_file.write("Trigram Validation Size :: " + str(validation_set_size) +
                          " Perplexity :: " + str(trigram_perplexity))
        output_file.write("\n\n")
        # sent_with_perplexity = dict()
        # for i in range(5):
        #     sent = trigram_model.generate_sentence(min_length=10)
        #     perplexity = trigram_model.calculate_trigram_perplexity(sent)
        #     sent_with_perplexity[sent] = perplexity
        # print(sent_with_perplexity)
        #
        mixgram_model = MixgramModel([0.1, 0.3, 0.6], [unigram_model, bigram_model, trigram_model])
        mixgram_perplexity = mixgram_model.calculate_mixgram_perplexity(validation_sentences)
        print("Mixgram perplexity :: " + str(mixgram_perplexity))
        output_file.write("Mixgram Validation Size :: " + str(validation_set_size) +
                          " Perplexity :: " + str(mixgram_perplexity))
        output_file.write("\n\n")
        # sent_with_perplexity = dict()
        # for i in range(5):
        #     sent = mixgram_model.generate_sentence(min_length=10)
        #     perplexity = mixgram_model.calculate_mixgram_perplexity(sent)
        #     sent_with_perplexity[sent] = perplexity
        # print(sent_with_perplexity)