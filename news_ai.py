import sqlite3

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

database = "D:/Yang/projects/NewsAi/data.db"

interesting = []
not_interesting = []
dataset = []

with sqlite3.connect(database) as db:
    cursor = db.cursor()

    for value in cursor.execute("""SELECT "Name" FROM "News" WHERE "Exit" = ?""", ("Interesting",)):
        interesting.append(value[0])

    for value in cursor.execute("""SELECT "Name" FROM "News" WHERE "Exit" = ?""", ("Not interesting",)):
        not_interesting.append(value[0])

BOT_CONFIG = {"intents": {"Interesting": {"examples": interesting},
                          "Not interesting": {"examples": not_interesting}}}

for intent, intent_value in BOT_CONFIG["intents"].items():
    for example in intent_value["examples"]:
        dataset.append([example, intent])

examples = [example for example, intent in dataset]
intent = [intent for example, intent in dataset]


def get_prediction(name):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(examples)

    clf = LogisticRegression(random_state=0)
    clf.fit(X, intent)

    predict = clf.predict(vectorizer.transform([name]))

    predict_proba = clf.predict_proba(vectorizer.transform([name]))

    f = predict_proba[0][0]
    f = f * 100
    f = round(f, 2)
    print(f"Интересность = {f}%")
    return predict[0], f


def get_validate(x, y):
    scores = []
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(x)
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        clf = LogisticRegression(random_state=0)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores.append(score)

    validate = sum(scores) / len(scores)
    validate = validate * 100
    validate = round(validate, 2)
    print(f"Валидность = {validate}%")
    return validate


get_prediction("Россияне сократили число покупок в категории товаров для домашних животных на 15%")
get_validate(examples, intent)