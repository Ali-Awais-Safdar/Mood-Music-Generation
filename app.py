from flask import Flask, render_template, request, redirect, url_for
from model_functions import emotion, map_emotion_to_mood, generate
import os
import pygame

app = Flask(__name__)

generated_file_path = None  # Global variable to store the generated file path

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/generate_music', methods=['POST'])
def generate_music():
    global generated_file_path
    mapped_mood = None

    choice = int(request.form['choice'])

    if choice == 1:
        user_text = request.form['user_text']
        emotion_labels = emotion(user_text)[0]['label']
        mapped_mood = map_emotion_to_mood(emotion_labels)
        generated_file_path = generate(mapped_mood)
    elif choice == 2:
        mood = request.form['predefined_mood']
        mapped_mood = mood
        generated_file_path = generate(mood)

    return redirect(url_for('play_music', mapped_mood=mapped_mood))

@app.route('/play_music')
def play_music():
    mapped_mood = request.args.get('mapped_mood')
    return render_template('audio_player.html', mapped_mood=mapped_mood)

@app.route('/start_music')
def start_music():
    global generated_file_path

    if generated_file_path and os.path.exists(generated_file_path):
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(generated_file_path)
        pygame.mixer.music.play()

    return "Music started."

@app.route('/stop_music')
def stop_music():
    pygame.mixer.music.stop()
    return "Music stopped."

if __name__ == '__main__':
    app.run(debug=True)