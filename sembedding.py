import fasttext
def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

model = fasttext.load_model('cc.en.300.bin')
print(len(model.words), type(model.words))

model.quantize()

# then display results and save the new model :
print_results(*model.test(valid_data))
model.save_model("model_filename.ftz")
