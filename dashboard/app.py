# dashboard/app.py

import dash
from dash import html, dcc, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import joblib

# Load the trained Linear Regression model
model = joblib.load("../models/linear_regression_model.pkl")

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # for deployment

# App layout
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("CartGenius Weight Prediction Dashboard", className="text-center text-primary mb-4"),
                width=12
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Enter Height (in inches):", className="fw-bold"),
                        dcc.Input(id="height-input", type="number", placeholder="e.g., 70", value=65),
                        html.Br(),
                        html.Br(),
                        dbc.Button("Predict Weight", id="predict-btn", color="primary")
                    ],
                    width=4
                ),
                dbc.Col(
                    [
                        html.H4("Prediction Result:"),
                        html.Div(id="prediction-output", className="fs-4 text-success")
                    ],
                    width=8
                )
            ]
        )
    ],
    fluid=True,
    className="p-4"
)

# Callback for prediction
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    Input("height-input", "value")
)
def predict_weight(n_clicks, height):
    if height is None:
        return "Please enter a height value."
    try:
        df = pd.DataFrame([[height]], columns=["Height"])
        weight = model.predict(df)[0]
        return f"Predicted Weight: {weight:.2f} lbs"
    except Exception as e:
        return f"Error in prediction: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)

