from torch.utils.data import Dataset
import os
import csv
from utils import remove_urls

class UbuntuCorpus(Dataset):
	def __init__(self, tokenizer, dir='./dialogs', max_seq_len=512, _cnt=30):
		super().__init__()
		dialogs = []
		thr = 30

		qa_pairs = []

		for subdir in os.listdir(dir):
			for dialog in os.listdir(dir + '/' + subdir):
				path = dir + '/' + subdir +'/' + dialog
				with open(path) as tsvfile:
					reader = csv.reader(tsvfile, delimiter='\t')
					rows = [(row[1], row[-1]) for row in reader]
					replicas = []
					authors = set()
					author = -1
					for row in rows:
						if author == row[0]:
							replicas[-1].append(row[1])
						else:
							author = row[0]
							authors.add(author)
							replicas.append([row[1]])
					assert(len(authors) <= 2)
				'''
				Answer replic is a replic without ?
			 	Question replic is a replic with ? followed by answer replic

			 	Both must be longer than thr (after link replacemenets)

				And due to BERT restrictions both in tokenized form must be shorter than max_seq_len
				'''

				for i in range(len(replicas)):
					replicas[i] = '[CLS] ' + remove_urls(' '.join(replicas[i]))

				for i in range(len(replicas) - 1):
					if replicas[i].count('?') > 0 and replicas[i + 1].count('?') == 0 \
					  and min(len(replicas[i]), len(replicas[i + 1])) >= thr \
					  and len(tokenizer.tokenize(replicas[i])) <= max_seq_len \
					  and len(tokenizer.tokenize(replicas[i + 1])) <= max_seq_len:
						qa_pairs.append([replicas[i], replicas[i + 1]])
						_cnt -= 1
						if _cnt <=0:
							break
				if _cnt <= 0:
					break
			if _cnt <=0:
				break

		'''for el in qa_pairs:
		  print('>>', el[0])
		  print('>>>', el[1])
		  print()'''

		self.qa_pairs = qa_pairs

	def __len__(self):
		return len(self.qa_pairs)

	def __getitem__(self, ind):
		return (self.qa_pairs[ind][0], self.qa_pairs[ind][1])


class TwittCorpus(Dataset):
	def __init__(self, tokenizer, path='corp (1).txt'):
		'''
		Gets Path to TXT file in format
		[CLS] Qestion [SEP] \n
		[CLS] Answer [SEP]\n
		\n
		...
		'''
		super(twitt_dataset, self).__init__()
		with open(path, 'r') as f:
			reps = f.readlines()
		dgs = [list(group) for k, group in groupby(reps, lambda x: x == '\n') if not k]
		for i in range(len(dgs)):
			dgs[i][0] = dgs[i][0].remove('[SEP]', '').rstrip()
			dgs[i][1] = dgs[i][1].remove('[SEP]', '').rstrip()
		self.qa_s = dgs

	def __len__(self):
		return len(self.qa_s)

	def __getitem__(self, idx):
		return (self.qa_s[idx][0], self.qa_s[idx][1])
