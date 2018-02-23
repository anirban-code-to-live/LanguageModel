from src.bigram import BigramModel
from src.unigram import UnigramModel
from src.trigram import TrigramModel
from nltk import ngrams
from src.mixgram import MixgramModel
import numpy as np
from nltk.corpus import gutenberg
from nltk.corpus import brown


if __name__ == '__main__':

    sentences = gutenberg.sents()
    print(len(sentences))
    # print(sentences[0])
    words = gutenberg.words()

    validation_sentences = brown.sents()[0:200]
    print(len(validation_sentences))

    unigram_model = UnigramModel(words, smoothing=False)
    # prob = unigram_model.calculate_unigram_probability('Hello')
    # print(prob)
    # unigram_perplexity = unigram_model.calculate_unigram_perplexity(validation_sentences)
    # print("Unigram perplexity :: " + str(unigram_perplexity))
    #
    bigram_model = BigramModel(sentences, words, smoothing=False)
    # prob = bigram_model.calculate_bigram_probabilty('hello', 'hello')
    # # print(prob)
    # prob = bigram_model.cal_prob_good_turing_bigram("Hey", "Hello")
    # print(prob)
    # bigram_perplexity = bigram_model.calculate_bigram_perplexity(validation_sentences)
    # print("Bigram perplexity :: " + str(bigram_perplexity))
    #
    trigram_model = TrigramModel(sentences, words, smoothing=True)
    # prob = trigram_model.calculate_trigram_probabilty("hey", "there", "the")
    # trigram_perplexity = trigram_model.calculate_trigram_perplexity(validation_sentences)
    # print("Trigram perplexity :: " + str(trigram_perplexity))
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
    sent_with_perplexity = dict()
    for i in range(5):
        sent = mixgram_model.generate_sentence(min_length=10)
        perplexity = mixgram_model.calculate_mixgram_perplexity(sent)
        sent_with_perplexity[sent] = perplexity
    print(sent_with_perplexity)



