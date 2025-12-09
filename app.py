from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Initialize Flask app
app = Flask(__name__)

# Home page route
@app.route("/")
def home():
    # Load CSV data
    df = pd.read_csv("data/user_metrics.csv")

    # User Analytics Data
    user_data = df['DAU'].tolist()

    # Predict next month DAU using Linear Regression
    months = np.array(range(len(user_data))).reshape(-1, 1)
    target = np.array(user_data)
    model = LinearRegression()
    model.fit(months, target)
    next_month_pred = round(model.predict([[len(user_data)]])[0])
    predicted_user_data = user_data + [next_month_pred]  # Append prediction

    # Feature Insights Data
    feature_columns = ['Feature1','Feature2','Feature3']
    feature_data = df[feature_columns].sum().tolist()

    # Customer Segmentation Data (latest month)
    segment_columns = ['SegmentHigh','SegmentMedium','SegmentLow']
    segment_data = df[segment_columns].iloc[-1].tolist()

    # Pass data to template
    return render_template("index.html",
                           user_data=predicted_user_data,  # includes predicted next month
                           feature_data=feature_data,
                           segment_data=segment_data)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)


