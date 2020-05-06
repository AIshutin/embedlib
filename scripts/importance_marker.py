import embedlib
import argparse
import torch
import json
import dashtable
import logging
import string
import emoji
import torch.nn.functional as F
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

model = None

digits = "0123456789"
punctuation = string.punctuation + '’“”’—‘' + '–'

def char_is_emoji(character):
    return character in emoji.UNICODE_EMOJI

def parse_to_words(text):
    text = text.lower()
    words = []
    words_indexes = []
    current = []
    for i, el in enumerate(text + ' '):
        if el.isalpha():
            current.append(el)
        elif el in digits:
            if len(current) != 0 and not current[-1] in digits:
                words.append(''.join(current))
                words_indexes.append([i - len(current), i - 1])
                current = []
            current.append(el)
        elif el.isspace():
            if len(current) != 0:
                words.append(''.join(current))
                words_indexes.append([i - len(current), i - 1])
                current = []
        elif el in punctuation: # commas, dots, slashes
            if len(current) != 0:
                words.append(''.join(current))
                words_indexes.append([i - len(current), i - 1])
                current = []
            words.append(el)
            words_indexes.append([i, i])
        elif char_is_emoji(el): # emoji
            words.append('[UNK]')
            words_indexes.append([i, i])
        else:
            logging.warning(f"Warning! Strange char {el}")
    return words, words_indexes

def text_processer(text, words, words_indexes):
    original = model(text)
    embeddings = []
    for i in range(len(words)):
        tag = nltk.pos_tag([words[i]])[0][1]
        logging.debug(f"words[i]: {tag}")
        if pos2pronouns.get(tag, None) is None:
            placeholder = '[MASK]'
        else:
            placeholder = pos2pronouns.get(tag, None)
        text2 = text[:words_indexes[i][0]]
        text2 = text2 + placeholder + text[words_indexes[i][1] + 1:]
        embeddings.append(model(text2))
    return original, embeddings

def calc_self_importance(token_gradients, match, words=[]):
    grads = [0] * (len(match) - 1)
    for i in range(1, len(match)):
        for j in range(token_gradients.shape[1]):
            grads[match[i]] += token_gradients[0][j][i].item()**2
    total = sum(grads)
    grads = [el / total for el in grads]
    return grads

def extract_top_n_important_words(text, n=5):
    words, words_indexes = parse_to_words(text)
    original, embeddings = text_processer(text, words, words_indexes)
    results = [((embedding - original) ** 2).sum() for embedding in embeddings]
    rating = [[results[i], words[i], words_indexes[i]] for i in range(len(words))]
    rating.sort(reverse=True)
    return rating[:n]

pos2pronouns_en = {}
'''
    'ADJ': 'some',
    'RB': 'somehow',
    'RBR': 'somehow',
    'RBS': 'somehow',
    'NN': 'something',
    'NUM': '145',
    'PRON': 'someone',
    'VB': 'do',
    'VBD': 'did',
    'VBG': 'doing',
    'VBN': 'done',
    'VBP': 'do',
    'VBZ': 'does',
    'WDT': '[MASK]',
    'WP': '[MASK]',
    'WP$': '[MASK]',
    'WRB': '[MASK]'
}'''

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

LANGUAGE = 'english'
pos2pronouns = {'english': pos2pronouns_en}[LANGUAGE]

INF = int(1e10)

class SumTopK:
    def __init__(self, k):
        self.k = k
    def __call__(self, data):
        data.sort(reverse=True)
        return sum(data[:self.k])

def extract_top_n_important_sentences(text, n=5, merge_function=SumTopK(5)):
    tokenizer = nltk.data.load(f'tokenizers/punkt/{LANGUAGE}.pickle')
    sentences_with_commas = tokenizer.tokenize(text)
    sentences = []
    for el in sentences_with_commas:
        sentences += list(el.split(','))
    word_rating = extract_top_n_important_words(text, n=INF)
    sentences_indexes = []
    prev = 0
    for el in sentences:
        ind = text.find(el)
        assert(ind != -1)
        sentences_indexes.append([prev + ind, prev + ind + len(el) - 1])
        prev = sentences_indexes[-1][-1] + 1
        text = text[ind + len(el):]

    sentence_rating = [[] for i in range(len(sentences))]
    for importance, word, (l, r) in word_rating:
        cnt = 0
        if word in punctuation and word != '?':
            continue
        for i, (l_sent, r_sent) in enumerate(sentences_indexes):
            if not (r < l_sent or r_sent < l):
                sentence_rating[i].append(importance)
                cnt += 1
        if cnt != 1:
            assert(cnt == 1)
    for i in range(len(sentence_rating)):
        sentence_rating[i] = merge_function(sentence_rating[i])

    div = sum(sentence_rating)
    for i in range(len(sentences)):
        sentence_rating[i] = (sentence_rating[i] / div, sentences[i], sentences_indexes[i])
    sentence_rating.sort(reverse=True)
    return sentence_rating[:n]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility to detect importance '
                                                   'of words in text for understanding')
    parser.add_argument('--text', help='text to process')
    parser.add_argument('--checkpoint', help='path to checkpoint of model to use')
    parser.add_argument('--v', help='model version')
    parser.add_argument('--n', default=3, type=int, help='number of words to mark')
    args = parser.parse_args()

    model = embedlib.Embedder(args.v, args.checkpoint)# embedlib.utils.load_model(args.checkpoint)
    texts = json.load(open(args.text))['text']
    for text in texts:
        words = extract_top_n_important_words(text, args.n)
        colorful_segments = sorted([el[-1] for el in words], reverse=True)
        for el in colorful_segments:
            text = text[:el[0]] + bcolors.OKGREEN + text[el[0]:el[1] + 1] + bcolors.ENDC + text[el[1] + 1:]
        print(text)
        print()
