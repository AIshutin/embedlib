from embedlib import similarity
import embedlib
model = embedlib.Embedder('???', '../laser-3lays-mrr0.67')
# "../rubert-base-uncased")#"../ru-6-attentions")

pizza = "Я хочу заказать пиццу"
hello = "Привет"
sushi = "Я хочу заказать суши"
order = "Как заказать пиццу?"
garbage = "Библиотека ночью закрыта." #"Самураи точат мечи" #"Библиотека ночью закрыта."

print(f"'{order}' sim score with '{pizza}'")
print(similarity(model(order), model(pizza, True))[0][0])

print(f"'{order}' sim score with '{sushi}'")
print(similarity(model(order), model(sushi, True))[0][0])

print(f"'{order}' sim score with '{hello}'")
print(similarity(model(order), model(hello, True))[0][0])

print(f"'{order}' sim score with '{garbage}'")
print(similarity(model(order), model(garbage, True))[0][0])
