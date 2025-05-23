from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load necessary files
area_categories = joblib.load('area_categories.pkl')
facing_categories = joblib.load('facing_categories.pkl')
model = joblib.load('house_price_model.pkl')
X_columns = joblib.load('X_columns.pkl')  # list of model input columns

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None  # default
    if request.method == 'POST':
        # Get form data
        input_data = {
            'type_area': request.form['type_area'],
            'value_area': float(request.form['value_area']),
            'floor': int(request.form['floor']),
            'transaction': request.form['transaction'],
            'furnishing': request.form['furnishing'],
            'facing': request.form['facing'],
            'price_sqft': float(request.form['price_sqft']),
            'area': request.form['area'],
            'bhk': int(request.form['bhk']),
            'months_until_possession': int(request.form['months_until_possession'])
        }

        # Convert & preprocess input_data
        df = pd.DataFrame([input_data])

        # Mapping for categorical values
        type_area_map = {'Carpet Area': 0, 'Super Area': 1}
        furnishing_map = {'Unfurnished': 0, 'Semi Furnished': 1, 'Furnished': 2}
        transaction_map = {'New Property': 1, 'Resale': 0}

        df['type_area'] = df['type_area'].map(type_area_map)
        df['furnishing'] = df['furnishing'].map(furnishing_map)
        df['transaction'] = df['transaction'].map(transaction_map)

        df = pd.get_dummies(df, columns=['area', 'facing'])

        # Align with training data
        df = df.reindex(columns=X_columns, fill_value=0)

        # Predict
        prediction = model.predict(df)[0]
        prediction = round(prediction, 2)

    return render_template(
        'index.html',
        prediction=prediction,
        area_categories=area_categories,
        facing_categories=facing_categories
    )

if __name__ == '__main__':
    app.run(debug=True)
