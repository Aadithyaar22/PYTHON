from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get form data
        age = float(request.form['age'])
        income = float(request.form['income'])
        genre = int(request.form['genre'])  # 0 for Male, 1 for Female
        
        # Predict
        features = np.array([[age, income, genre]])
        prediction = model.predict(features)[0]
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)