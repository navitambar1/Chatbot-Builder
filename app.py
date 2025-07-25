from flask import Flask, request, render_template
import os
from chatbot_engine import process_file_and_embed, answer_from_bot
from flask import jsonify

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/create_bot', methods=['POST'])
def create_bot():
    name = request.form['user_name']
    brand = request.form['brand_name']
    color = request.form['brand_color']
    prompt = request.form['bot_prompt']
    file = request.files['training_file']
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    bot_id = f"{brand.lower().replace(' ', '_')}"
    process_file_and_embed(file_path, bot_id, prompt)

    return f"Bot Created! Access at /chat/{bot_id}"


# Store prompt & color info per bot in memory (could use DB later)
bot_registry = {}

@app.route('/chat/<bot_id>', methods=['GET', 'POST'])
def chat(bot_id):
    if request.method == 'GET':
        # For now, hardcoding brand info
        return render_template('chat.html', brand=bot_id.replace('_', ' ').title(), color="#1976d2")

    if request.method == 'POST':
        user_msg = request.json['message']
        reply = answer_from_bot(bot_id, user_msg)
        return jsonify({'reply': reply})


if __name__ == '__main__':
    app.run(debug=True, host="localhost", port=5000)
