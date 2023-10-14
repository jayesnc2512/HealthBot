from flask import Flask, render_template, request,jsonify
from chat_test import chatbot_logic
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")  # Create an HTML template for your chat interface

@app.route("/chat", methods=["POST"])
def chat():
    user_responses = request.form.to_dict()
    chatbot_responses = chatbot_logic(user_responses)
    return jsonify({"chatbot_responses": chatbot_responses})

if __name__ == "__main__":
    app.run(debug=True)