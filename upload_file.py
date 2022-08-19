import json, requests

filename = input('Путь к файлу json: ')
list_product = []
count_request = 20
filename.replace('\\', '/')

with open(filename, encoding='utf-8') as file:
    file_to_load = json.load(file)

for i in range(len(file_to_load) // count_request):
    list_product = [val for val in file_to_load[i*count_request:i*count_request+count_request]]
    response = requests.get(f'http://127.0.0.1:5000/predict?user_request={list_product}')
    print(response.content.decode('utf-8'))

if len(file_to_load) % count_request != 0:
    list_product = [val for val in file_to_load[len(file_to_load) // count_request * count_request:]]
    response = requests.get(f'http://127.0.0.1:5000/predict?user_request={list_product}')
    print(response.content.decode('utf-8'))