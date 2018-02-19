from src import tokenization
from src import unigram
from src import bigram
from src import trigram
from src import kneser_ney
from nltk import ngrams
from src import mixgram
import numpy as np
from nltk.corpus import gutenberg
from nltk.corpus import brown


if __name__ == '__main__':
    # example_text = "Hello Pikachu, How are you doing? Have you had your breakfast, today? " \
    #                "Are you going to eat Dosa?" \
    #                "I love it."
    #
    # sentences = tokenization.tokenize_sentence(example_text)
    # # print(sentences)
    # words = tokenization.tokenize_words(example_text)
    # # print(len(words))
    # unique_words = np.unique(words)
    # # print(len(unique_words))
    # filtered_sentence, stop_words = tokenization.tokenize_words_without_stopwords(example_text)
    # # print(filtered_sentence)
    # stemmed_tokens = tokenization.stem_tokens(example_text)
    # # print(stemmed_tokens)
    # parts_of_speech_tagged_words = tokenization.tag_parts_of_speech(example_text, example_text)
    # # print(parts_of_speech_tagged_words)

    sentences = gutenberg.sents()
    print(len(sentences))
    # print(sentences[0])
    words = gutenberg.words()

    validation_sentences = brown.sents()[0:5000]
    print(len(validation_sentences))

    unigram_model = unigram.UnigramModel(words, smoothing=False)
    # prob = unigram_model.calculate_unigram_probability('Hello')
    # # print(prob)
    # unigram_perplexity = unigram_model.calculate_unigram_perplexity(validation_sentences)
    # print("Unigram perplexity :: " + str(unigram_perplexity))
    #
    bigram_model = bigram.BigramModel(sentences, words, smoothing=False)
    # prob = bigram_model.calculate_bigram_probabilty('hello', 'hello')
    # # print(prob)
    # bigram_perplexity = bigram_model.calculate_bigram_perplexity(validation_sentences)
    # print("Bigram perplexity :: " + str(bigram_perplexity))
    #
    trigram_model = trigram.TrigramModel(sentences, words, smoothing=False)
    # prob = trigram_model.calculate_trigram_probabilty("hey", "there", "the")
    # trigram_perplexity = trigram_model.calculate_trigram_perplexity(validation_sentences)
    # print("Trigram perplexity :: " + str(trigram_perplexity))
    #
    mixgram_model = mixgram.MixgramModel([0.2, 0.7, 0.1], [unigram_model, bigram_model, trigram_model])
    mixgram_perplexity = mixgram_model.calculate_mixgram_perplexity(validation_sentences)
    print("Mixgram perplexity :: " + str(mixgram_perplexity))
    sent_with_perplexity = dict()
    for i in range(4):
        sent = mixgram_model.generate_sentence(min_length=10)
        perplexity = mixgram_model.calculate_mixgram_perplexity(sent)
        sent_with_perplexity[sent] = perplexity
    print(sent_with_perplexity)

    # gut_ngrams = (
    #     ngram for sent in gutenberg.sents() for ngram in ngrams(sent, 3,
    #                                                             pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
    #
    # kneser_ney_model = kneser_ney.KneserNeyLM(3, gut_ngrams, start_pad_symbol='<s>', end_pad_symbol='</s>')
    #
    # # lm.score_sent(('This', 'is', 'a', 'sample', 'sentence', '.'))
    # print(kneser_ney_model.calculate_kneser_ney_perplexity(validation_sentences))
    # sent_with_perplexity = dict()
    # for i in range(3):
    #     sent = kneser_ney_model.generate_sentence(min_length=10)
    #     perplexity = kneser_ney_model.calculate_kneser_ney_perplexity(sent)
    #     sent_with_perplexity[sent] = perplexity
    # print(sent_with_perplexity)



