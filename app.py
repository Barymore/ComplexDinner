'''
Онлайн-хакатон от AGORA при поддержке Акселератора Возможностей при ИНТЦ МГУ «Воробьевы горы»
Программа: "Разработка системы с web-интерфейсом для сопоставления характеристик товаров маркетплейса с их эталонными значениями"
Команда: ComplexDinner

Online hackathon from AGORA with the support of the Accelerator of Opportunities at the INTC MSU "Vorobyovy Gory"
Program: "Development of a system with a web interface for the corresponding characteristics of the marketplace products with their reference values"
Team: Complex Dinner
'''


'''
Импорт библиотек:
flask - для работы API
sklearn - для машинного обучения
tokenize - лексический сканер


Library import:
flask - for API work
sklearn - for machine learning
tokenize - lexical scanner
'''

from flask import Flask, request, jsonify
import json
import os
import re
import pickle
from tokenize import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__) 
JSON_PATH = r"C:\Users\somow\Documents\Lesa\Text-Classification-using-random-forest-classifier\agora_hack_products.json"
FORMAT = ".pickle"
LABEL_ENCODER_NAME = LabelEncoder.__name__ + FORMAT
TFIDF_VECTORIZER_NAME = TfidfVectorizer.__name__ + FORMAT
CLASSIFIER_NAME = r'C:\Users\somow\Documents\Lesa\Text-Classification-using-random-forest-classifier\RandomForestClassifier.pickle'
vectorizer = TfidfVectorizer(max_features=2500)


def json_reader(json_path=JSON_PATH):
    ''' 
    Фунция для чтения .json файла для обучения модели, аргумент: путь к файлу
    Function for reading .json file for model training, argument: file path
    '''
    try:
        with open(json_path, "r", encoding="utf-8-sig") as js:
            return json.load(js)
    except FileNotFoundError as ex:
        print(ex)


def pkl_writer(filename=CLASSIFIER_NAME, to_dump=None, format=FORMAT):
    '''
    Функция для записи весов модели в файл .pickle, аргументы: названия файла,  объект для записи, формат записи
    Function for writing model weights to a .pickle file, arguments: file name, object to write, write format
    '''
    if not filename.endswith(format):
        filename = filename + format
    with open(filename, 'wb') as handle:
        pickle.dump(to_dump, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(filename, "was writed")


def pkl_reader(filename):
    '''
    Функция для чтения весов модели из файла .pickle, аргумент: путь к файлу
    Function to read model weights from .pickle file, argument: file path
    '''
    print(filename)
    with open(filename, 'rb') as pkl:
        return pickle.load(pkl)


def contents_preparation(js):
    '''
    Функция для обработки файла .json, аогумент: путь к файлу .json
    Function to process .json file, argument: path to .json file
    '''
    contents_for_predict, ref_ids = [], []
    for j in js:
        try:
            is_ref = j["is_reference"] 
            ref_id = j["reference_id"] if not is_ref else j["product_id"]  
            ref_ids.append(ref_id)  
        except Exception as ex:
            pass
        content = j["name"]
        content = re.sub(r'\W', " ", content)
        contents_for_predict.append(content)
    return contents_for_predict, ref_ids

def make_classifier(X_train, X_test, y_train, y_test):
    '''
    Функция для создания модели машинного обучения
    Function to create a machine learning model
    '''
    classifier = RandomForestClassifier(n_estimators=350, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("accuracy score:", metrics.accuracy_score(y_pred, y_test))
    pkl_writer(filename=CLASSIFIER_NAME, to_dump=classifier)
    return classifier


def cold_start():
    '''
    Функция для первого запуска
    Function for the first run
    '''
    js = json_reader()
    contents_for_predict, ref_ids = contents_preparation(js=js)
    train_text = contents_for_predict

    train_y = label_encoder.transform(ref_ids)
    X_train = tf_idf_vectorizer.transform(train_text).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X_train, train_y, test_size=0.3, random_state=0)
    classifier = make_classifier(X_train, X_test, y_train, y_test)
    return classifier


def get_model(classifier_name=CLASSIFIER_NAME):
    '''
    Функция для прочтения модели из файла, аргумент: путь к файлу
    Function for reading the model from a file, argument: file path
    '''
    if not os.path.exists(classifier_name):
        print(os.path.exists(classifier_name), classifier_name)
        classifier = cold_start()
    else:
        classifier = pkl_reader(classifier_name)[0]
    return classifier


def make_prediction(js, classifier, label_encoder):
    '''
    Функция для выполнения предсказания
    Function to perform prediction
    '''
    for_predict, ids = contents_preparation(js=js)
    for_predict = tf_idf_vectorizer.transform(
             for_predict).toarray()
    proba = classifier.predict_proba(for_predict)
    threshold = 0.8
    cls = [c.argmax() for c in proba]
    prob = [c.max() for c in proba]
    result = []
    for c, p in zip(cls, prob):
        res = c if p > threshold else -1
        print(res)
        try:
            result.append(label_encoder.inverse_transform([res])[-1])
        except ValueError:
            result.append(-1)
    qwe = list(map(lambda x: x if x != -1 else "no label", result))
    print(result, c, p)
    return qwe



@app.route('/predict', methods=['POST'])
def predict_for_string():
    '''
    Функция API для предсказания эталона
    API function for benchmark prediction
    '''
    if request.method == 'POST':
        user_request = request.json
        user_request = json.loads(user_request)['tasks']
        predicted_data = make_prediction(js=user_request, classifier=classifier, label_encoder=label_encoder)
        for i in range(len(user_request)):
            user_request[i]['predicted_id_preference'] = predicted_data[i]
        response = json.dumps(user_request)
        return response

tf_idf_vectorizer = vectorizer.fit(contents_preparation(json_reader(JSON_PATH))[0])
la_enc = LabelEncoder()
label_encoder = la_enc.fit(contents_preparation(json_reader(JSON_PATH))[1])
classifier = get_model(classifier_name=CLASSIFIER_NAME)

if __name__ == '__main__':
    app.run(port=5000)
