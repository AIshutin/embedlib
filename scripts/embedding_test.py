from bert_serving.client import BertClient
bc = BertClient()
res = list(bc.encode(['First do it']))
for el in res:
	print(el)
print()
