import embedlib
import sys

def evaluate_task(quest, answs, silent=False):
    if not silent:
        print(quest)
    scores = []
    quest = model(quest)
    for i in range(len(answs)):
        answ_text = answs[i]
        answ = model(answ_text, is_question=False)
        score = embedlib.similarity(quest, answ)[0][0]
        if not silent:
            print(answ_text)
            print(score)
        scores.append([score, i])
    scores = sorted(scores, reverse=True)
    errors = 0
    for i in range(len(scores)):
        errors += (i - scores[i][1]) ** 2
    if not silent:
        print()
    return errors

model_name = 'laser-3lays-transformer-mrr0.8' # '../ru-1-attention-finetuned' "../rubert-base-uncased"
model = embedlib.Embedder('???', f"{model_name}")

total_erros = 0
with open(sys.argv[1], encoding="utf-8") as file:
    quest = None
    answs = []
    token = '[Q] '
    for line in file.readlines():
        line = line.strip()
        if token in line:
            answs = []
            quest = line[len(token):]
        elif len(line) == 0:
            total_erros += evaluate_task(quest, answs)
        else:
            answs.append(line[len(token):])

    total_erros += evaluate_task(quest, answs)
print(f"{total_erros} wrong score")
