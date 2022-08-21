# ComplexDinner Agora Hack Кейс №1

Для запуска:
$ flask run
  * Running on localhost:5000/ (Press CTRL+C to quit)


Формат запроса:
Запрос
POST localhost:5000/predict

Тело запроса - JSON
{'tasks' : [
  {
    "id": "some_id_1",
    "name": "Название товара1",
    "props": [...]
  },
  {
    "id": "some_id_2",
    "name": "Название товара2",
    "props": [...]
  },
  {
    "id": "some_id_3",
    "name": "Название товара3",
    "props": [...]
  }
]
}

Тело ответа
[
  {
    "id": "some_id_1",
    "name": "Название товара1",
    "props": [...],
    "predicted_id_preference": "reference_1_id"
  },
  {
    "id": "some_id_2",
    "name": "Название товара2",
    "props": [...],
    "predicted_id_preference": "reference_2_id"
  },
  {
    "id": "some_id_3",
    "name": "Название товара3",
    "props": [...],
    "predicted_id_preference": "reference_3_id"
  }
]