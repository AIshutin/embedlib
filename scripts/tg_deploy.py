import embedlib
import torch
import gc
import transformers
import time
import telegram
import logging
from time import sleep
from telegram.error import NetworkError, Unauthorized

model = embedlib.models.GPT2Like('ru')
model, tokenizer = model.qembedder, model.tokenizer
model = transformers.GPT2LMHeadModel.from_pretrained('../ru-GPT2Like')
device = torch.device('cuda:0')
model.to(device)
max_add_length = 70
EOS_ID = 50000

def make_response(hist):
    input_ids = []
    for el in hist:
        input_ids += el
    start_len = len(input_ids)
    input_ids = torch.tensor(input_ids, device=device, dtype=torch.long).unsqueeze(0)
    beam_output = model.generate(
        input_ids,
        max_length=len(input_ids) + max_add_length,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2
    )

    generated = list(beam_output[0].cpu())[start_len:]
    for i in range(len(generated)):
        if generated[i] == EOS_ID:
            return generated[:i]
    return generated

TOKEN = "873557022:AAH79Kyuwk3ZoNITS4_H5Keku7Q4pQInpaQ"
uid2hist = {}
update_id = None
LAST_REPLIC = 5

def echo(bot):
    """Echo the message the user sent."""
    global update_id
    # Request updates after the last update_id
    for update in bot.get_updates(offset=update_id, timeout=30):
        update_id = update.update_id + 1

        if update.message:  # your bot can receive updates without messages
            # Reply to the message'
            logging.info(update.message)
            from_id = update.message.from_user.id
            if from_id not in uid2hist:
                uid2hist[from_id] = []
            if update.message.text == '/restart':
                uid2hist[from_id] = []
                continue
            uid2hist[from_id].append(tokenizer.encode(update.message.text) + [EOS_ID])
            response = make_response(uid2hist[from_id][-LAST_REPLIC:])
            text = tokenizer.decode(response, skip_special_tokens=True)
            uid2hist[from_id].append(response + [EOS_ID])
            update.message.reply_text(text)


if __name__ == '__main__':
    bot = telegram.Bot(TOKEN)
    try:
        update_id = bot.get_updates()[0].update_id
    except IndexError:
        update_id = None

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    while True:
        try:
            echo(bot)
        except NetworkError:
            sleep(1)
        except Unauthorized:
            # The user has removed or blocked the bot.
            update_id += 1



""" CLI:
hist = []
while True:
    curr = input('YOU>>')
    hist.append(tokenizer.encode(curr) + [EOS_ID])
    response = make_response(hist[-LAST_REPLIC:])
    print(tokenizer.decode(response, skip_special_tokens=True))
    hist.append(response + [EOS_ID])
"""
