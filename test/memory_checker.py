from embedlib.utils import load_model
from embedlib.utils import mem_report
import embedlib
import sys
import time
import json
import argparse
import os

parser = argparse.ArgumentParser(description="utility to run model on several batches "
                                            "to measure memory consumption")
parser.add_argument('--text', help="path to json file to process. File format:"
                                   "{\"text\": [\"str1\", \"str2\"]}")
parser.add_argument('--checkpoint', help="checkpoint directory")
parser.add_argument('--batch_size', default=1)
parser.add_argument('--logfile', default='logs.txt')
args = parser.parse_args()
args.batch_size = int(args.batch_size)

with open(args.logfile, 'w') as file:
    text = json.load(open(args.text))['text']
    assert(len(text) >= args.batch_size)
    assert(len(text) < 1000 * args.batch_size) # too match

    model = embedlib.Embedder(json.load(open(os.path.join(args.checkpoint, 'model_config.json')))['version'],
                            args.checkpoint)
    for i in range(args.batch_size - 1, len(text)):
        batch = text[i - args.batch_size + 1:i]
        start = time.time()
        res = model(batch)
        end = time.time() - start
        print(f"Batch embedded in: {end:9.4f}", file=file)
