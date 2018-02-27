from src.bigram import BigramModel
from src.unigram import UnigramModel
from src.trigram import TrigramModel
from nltk import ngrams
import nltk
from src.mixgram import MixgramModel
import numpy as np
from nltk.corpus import gutenberg
from nltk.corpus import brown
import os
import random


if __name__ == '__main__':

    if not os.path.exists("../output_data"):
        os.makedirs("../output_data")

    nltk.download('punkt')
    nltk.download('gutenberg')
    nltk.download('brown')

    # Sentences generated from with Gutenberg corpus
    sentences_gutenberg = gutenberg.sents()
    unigram_model = UnigramModel(sentences_gutenberg, smoothing="GoodTuring")
    bigram_model = BigramModel(sentences_gutenberg, smoothing="GoodTuring")
    trigram_model = TrigramModel(sentences_gutenberg, smoothing="GoodTuring")
    mixgram_model = MixgramModel([0.2, 0.2, 0.6], [unigram_model, bigram_model, trigram_model])

    sent_with_perplexity = dict()
    for i in range(3):
        sent = mixgram_model.generate_sentence(min_length=10)
        perplexity = mixgram_model.calculate_mixgram_perplexity(sent)
        sent_with_perplexity[sent] = perplexity
    random_sent = random.choice(list(sent_with_perplexity.keys()))
    print(random_sent)
    sent_file = open('sentence.txt', 'w')
    sent_file.write(str(random_sent))

    # output_file = open('../output_data/perplexity.txt', 'a')
    # Tune lambda values
    # sentences_gutenberg = gutenberg.sents()
    # sentences_brown = brown.sents()
    # merged_gutenberg_brown_sents = sentences_gutenberg + sentences_brown
    # train_set_size = int(0.6 * len(merged_gutenberg_brown_sents))
    # train_set_sents = merged_gutenberg_brown_sents[:train_set_size]
    # output_file.write("\n\n\n")
    # output_file.write("Train Corpus :: Gutenberg + Brown, Validation Corpus:: Brown \n")
    # validation_sentences = merged_gutenberg_brown_sents[-500:]
    # weights = [[0.1, 0.2, 0.7], [0.1, 0.4, 0.5], [0.1, 0.1, 0.8],
    #            [0.2, 0.2, 0.6], [0.2, 0.3, 0.5], [0.2, 0.4, 0.4]]
    #
    # for weight in weights:
    #     print(str(weight))
    #     unigram_model = UnigramModel(train_set_sents, smoothing="GoodTuring")
    #     bigram_model = BigramModel(train_set_sents, smoothing="GoodTuring")
    #     trigram_model = TrigramModel(train_set_sents, smoothing="GoodTuring")
    #     mixgram_model = MixgramModel(weight, [unigram_model, bigram_model, trigram_model])
    #     mixgram_perplexity = mixgram_model.calculate_mixgram_perplexity(validation_sentences)
    #     print("Mixgram perplexity :: " + str(mixgram_perplexity))
    #     output_file.write("Mixgram Weights :: " + str(weight) + "\n")
    #     output_file.write("Mixgram Validation Size :: " + str(500) +
    #                       " Perplexity :: " + str(mixgram_perplexity))
    #     output_file.write("\n\n")

    # Gutenberg + Brown
    # sentences_gutenberg = gutenberg.sents()
    # sentences_brown = brown.sents()
    # output_file.write("\n\n")
    # output_file.write("Train Corpus :: Gutenberg, Validation Corpus:: Brown \n")
    # validation_sentences = sentences_brown[0:500]
    #
    # for weight in weights:
    #     print(str(weight))
    #     unigram_model = UnigramModel(sentences_gutenberg, smoothing="GoodTuring")
    #     bigram_model = BigramModel(sentences_gutenberg, smoothing="GoodTuring")
    #     trigram_model = TrigramModel(sentences_gutenberg, smoothing="GoodTuring")
    #     mixgram_model = MixgramModel(weight, [unigram_model, bigram_model, trigram_model])
    #     mixgram_perplexity = mixgram_model.calculate_mixgram_perplexity(validation_sentences)
    #     print("Mixgram perplexity :: " + str(mixgram_perplexity))
    #     output_file.write("Mixgram Weights :: " + str(weight) + "\n")
    #     output_file.write("Mixgram Validation Size :: " + str(500) +
    #                       " Perplexity :: " + str(mixgram_perplexity))
    #     output_file.write("\n\n")
    #
    # # Brown + Gutenberg
    # sentences_gutenberg = gutenberg.sents()
    # sentences_brown = brown.sents()
    # output_file.write("\n\n")
    # output_file.write("Train Corpus :: Brown, Validation Corpus:: Gutenberg \n")
    # validation_sentences = sentences_gutenberg[0:500]
    #
    # for weight in weights:
    #     print(str(weight))
    #     unigram_model = UnigramModel(sentences_brown, smoothing="GoodTuring")
    #     bigram_model = BigramModel(sentences_brown, smoothing="GoodTuring")
    #     trigram_model = TrigramModel(sentences_brown, smoothing="GoodTuring")
    #     mixgram_model = MixgramModel(weight, [unigram_model, bigram_model, trigram_model])
    #     mixgram_perplexity = mixgram_model.calculate_mixgram_perplexity(validation_sentences)
    #     print("Mixgram perplexity :: " + str(mixgram_perplexity))
    #     output_file.write("Mixgram Weights :: " + str(weight) + "\n")
    #     output_file.write("Mixgram Validation Size :: " + str(500) +
    #                       " Perplexity :: " + str(mixgram_perplexity))
    #     output_file.write("\n\n")