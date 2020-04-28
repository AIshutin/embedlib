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

model = None

digits = "0123456789"
punctuation = string.punctuation + '’“”’—‘' + '–'

def char_is_emoji(character):
    return character in emoji.UNICODE_EMOJI

def match_tokens_with_words(tokens, text):
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
    assert(len(current) == 0)
    logging.debug('words', words)

    current_word = 0
    prefix = 0
    token2words = []
    for el in tokens:
        if prefix == 0:
            if el == words[current_word]:
                token2words.append(current_word)
                current_word += 1
            elif el == words[current_word][:len(el)]:
                token2words.append(current_word)
                prefix = len(el)
            else:
                logging.warning("Tokens to words can not be compleated. Returning current results")
                logging.warning(f"{el} {current_word} {words[current_word]} {prefix}")
                logging.warning(f"{match_tokens_with_words}")
                return None
        else:
            if '##' not in el:
                logging.warning(f"Expected '##' in {el} token")
            else:
                el = el[2:]
            if el == words[current_word][prefix:len(el) + prefix]:
                token2words.append(current_word)
                prefix += len(el)
                if prefix == len(words[current_word]):
                    prefix = 0
                    current_word += 1
            else:
                logging.warning(f"{el} {current_word} {words[current_word]} {prefix}")
                logging.warning(f"{match_tokens_with_words}")
                logging.warning("Tokens to words can not be compleated. Returning current results.")
                return None
    assert(len(words) == len(words_indexes))
    return [tokens, token2words, words, words_indexes]

def call_BERT_with_input_grad(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)

    # We create a 3D attention mask from a 2D tensor mask.
    # Sizes are [batch_size, 1, 1, to_seq_length]
    # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
    # this attention mask is more simple than the triangular masking of causal attention
    # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
    else:
        head_mask = [None] * self.config.num_hidden_layers

    embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
    embedding_output = torch.tensor(embedding_output, requires_grad=True)
    encoder_outputs = self.encoder(embedding_output,
                                   extended_attention_mask,
                                   head_mask=head_mask)
    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output)

    outputs = (sequence_output, pooled_output,) + encoder_outputs[1:] # add hidden_states and attentions if they are here
    return (outputs, embedding_output)

def text_processer(text):
    text = text[:model.max_seq_len]
    tokenized_text = model.tokenizer.encode(text, add_special_tokens=True)
    _, match, words, words_indexes = match_tokens_with_words(model.tokenizer.tokenize(text), text)
    tensor = torch.tensor([tokenized_text])
    textembedding, preembedding = call_BERT_with_input_grad(model.qembedder, tensor)
    embedding = F.normalize(model.get_embedding(textembedding[0]), dim=1)
    return embedding, preembedding, [None] + match, words, words_indexes

def calc_self_importance(token_gradients, match, words=[]):
    grads = [0] * (len(match) - 1)
    for i in range(1, len(match)):
        for j in range(token_gradients.shape[1]):
            grads[match[i]] += token_gradients[0][j][i].item()**2
    total = sum(grads)
    grads = [el / total for el in grads]
    return grads

def extract_top_n_important_words(text, text2=None, n=5):
    embedding, preembedding, match, words, words_indexes = text_processer(text)

    if text2 is None:
        embedding.pow(2).mean().backward()
    else:
        embedding2 = F.normalize(model.qembedd([text2]), dim=1)
        embedding2.dot(embedding).mean().backward()

    result = calc_self_importance(preembedding.grad, match, words)
    rating = [[result[i], words[i], words_indexes[i]] for i in range(len(words))]
    rating.sort(reverse=True)
    assert(abs(sum(result) - 1) < 0.1)
    return rating[:n]

LANGUAGE = 'english'
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
    print(sentence_rating)
    return sentence_rating[:n]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility to detect importance '
                                                   'of words in text for understanding')
    parser.add_argument('--text', help='text to process')
    parser.add_argument('--checkpoint', help='path to checkpoint of model to use')
    args = parser.parse_args()

    model = embedlib.utils.load_model(args.checkpoint)
    texts = json.load(open(args.text))['text']
    for text in texts:
        print(text)
        extract_top_n_important_sentences(text)

        print()
