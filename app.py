from flask import Flask, request, jsonify
import requests
from main import qa


app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def askchatbot():
    data = request.json
    prompt = data.get('prompt')
    if prompt:
        try:
            response = qa.invoke(prompt)
            return jsonify({"response": response}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "No prompt provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)