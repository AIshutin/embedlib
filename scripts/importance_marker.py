import embedlib
import argparse
import torch
import json
import dashtable
import logging
import string
import emoji

model = None

def char_is_emoji(character):
    return character in emoji.UNICODE_EMOJI

def match_tokens_with_words(tokens, text):
    text = text.lower()
    words = []
    digits = "0123456789"
    current = []
    punctuation = string.punctuation + '’“”’—‘' + '–'
    for el in text:
        if el.isalpha():
            current.append(el)
        elif el in digits:
            if len(current) != 0 and not current[-1] in digits:
                words.append(''.join(current))
                current = []
            current.append(el)
        elif el.isspace():
            if len(current) != 0:
                words.append(''.join(current))
                current = []
        elif el in punctuation: # commas, dots, slashes
            if len(current) != 0:
                words.append(''.join(current))
                current = []
            words.append(el)
        elif char_is_emoji(el): # emoji
            words.append('[UNK]')
        else:
            logging.warning(f"Warning! Strange char {el}")
    if len(current) != 0:
        words.append(''.join(current))
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

    return [tokens, token2words, words]

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
    tokenized_text = model.tokenizer.encode(text)
    assert(len(tokenized_text) == len(model.tokenizer.tokenize(text)))
    _, match, words = match_tokens_with_words(model.tokenizer.tokenize(text), text)
    tensor = torch.tensor([tokenized_text])
    textembedding, preembedding = call_BERT_with_input_grad(model.qembedder, tensor)
    embedding = model.get_embedding(textembedding[0])
    return (embedding, preembedding, match, words)

def calc_self_importance(token_gradients, match, words=[]):
    grads = [0] * len(words)
    for i in range(len(match)):
        for j in range(token_gradients.shape[1]):
            grads[match[i]] += token_gradients[0][j][i].item()**2
    total = sum(grads)
    grads = [el / total for el in grads]
    return grads

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility to detect importance '
                                                   'of words in text for understanding')
    parser.add_argument('--text', help='text to process')
    parser.add_argument('--checkpoint', help='path to checkpoint of model to use')
    args = parser.parse_args()

    model = embedlib.utils.load_model(args.checkpoint)
    texts = json.load(open(args.text))['text']
    for text in texts:
        embedding, preembedding, match, words = text_processer(text)
        embedding.pow(2).mean().backward()
        result = calc_self_importance(preembedding.grad, match, words)
        print(text)
        print(result)
        print(dashtable.data2md([words, result]))
        print()

# ToDo: word indexes in text
# ToDo: function to return top-N important words with their locations.
# ToDo: loss is dot product of the question and the answer.
