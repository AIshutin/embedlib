import embedlib

def leave_one_out(embedder, data, k=1):
    assert(len(data) > 1)
    X = []
    y = []
    for class_id in range(len(data)):
        #print(class_id)
        assert(len(data[class_id]) > 1)
        for el in data[class_id]:
            X.append(embedder(el))
            y.append(class_id)
    errors = 0
    for i in range(0, len(data)):
        best = -1
        best_dist = 1e9

        for j in range(len(data)):
            if i == j:
                continue
            dist = embedlib.similarity(X[i], X[j])
            if dist < best_dist:
                best = j
                best_dist = dist
        errors += y[i] != y[best]

    return (1 - errors / len(X))

def read_intents_file(filename):
    data = [[]]
    for line in open(filename, encoding='utf-8').readlines():
        line = line.rstrip()
        #print(line)
        if len(line) == 0:
            data.append([])
            #print('---')
        else:
            data[-1].append(line)
    if len(data[-1]) == 0:
        data.pop(-1)
    return data

def get_mean_knn_metric(embedder, data_filename=None):
    if data_filename is None:
        data_filename = f"{embedder.model.lang}-intents-data.txt"
    return leave_one_out(embedder, read_intents_file(data_filename))

if __name__ == "__main__":
    import sys
    import json

    model_folder = sys.argv[1]
    if model_folder[-1] != '/':
        model_folder = model_folder + '/'
    model_config = json.load(open(f'{model_folder}model_config.json'))
    #lang = model_config['lang']
    model_version = model_config['version']

    #intents_file = f'{lang}-intents-data.txt'

    embedder = embedlib.Embedder(model_version, model_folder)
    print(f"score: {get_mean_knn_metric(embedder):9.3f}")
