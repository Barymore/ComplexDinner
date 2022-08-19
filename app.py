from flask import Flask, request

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict_for_string():
    if request.method == 'GET':
        user_request = request.args.get('user_request')
        return f'ваш запрос = {user_request}'


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
