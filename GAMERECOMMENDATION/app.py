import pandas as pd
from skimage import io
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, render_template, request

data = pd.read_csv("gamePreferences.csv")
x = data.drop(columns=["game"])
y = data["game"]

model = DecisionTreeClassifier()
model.fit(x, y)

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def home():
    name = "valorant.jpg"
    if request.method == "POST":
        age = int(request.form["age"])
        sex = int(request.form["sex"])
        predicted_value = model.predict([[age, sex]])
        return render_template("index.html", predicted_game=predicted_value[0])
    return render_template("index.html", predicted_game=name)

if __name__ == "__main__":
    app.run(debug=True)
