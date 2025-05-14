from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
    print(f"Model type: {type(model)}")
    print(f"Model contents: {model}")  # Be careful with this if the model is large

# If the model is a dictionary, it might contain the actual model under a key
if isinstance(model, dict) and 'model' in model:
    model = model['model']
    print(f"Extracted model type: {type(model)}")

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
        
        # Check if model is callable or a dictionary with a callable under a key
        if hasattr(model, 'predict'):
            prediction = model.predict(features)[0]
        elif isinstance(model, dict):
            # Try common keys where a model might be stored
            for key in ['model', 'classifier', 'regressor', 'estimator']:
                if key in model and hasattr(model[key], 'predict'):
                    prediction = model[key].predict(features)[0]
                    break
            else:
                prediction = "Error: Could not find a valid model object"
        else:
            prediction = "Error: Model object doesn't have predict method"
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
