import embedlib

model = embedlib.Embedder(None, 'ru-1-attention-finetuned')
model('Привет')
