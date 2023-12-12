from flask import Flask, render_template, request, jsonify
import requests
from PIL import Image
from io import BytesIO
from fastai.vision.all import load_learner, PILImage

learn_inf = load_learner("weather_classifier.pkl")

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_prediction")
def get_prediction():
    """Return a prediction."""
    image_url = request.args.get("url")
    if image_url is None:
        return "No url provided", 400
    image_response = requests.get(image_url)
    if image_response.status_code != 200:
        return "Invalid url", 400
    try:
        pil_image = PILImage.create(BytesIO(image_response.content))
    except Exception as e:
        return f"Error loading image: {e}", 400
    try:
        pred, pred_idx, probs = learn_inf.predict(pil_image)
        print(pred)
        return jsonify({"prediction": pred, "probls": probs[pred_idx].item()})
    except Exception as e:
        return f"Error during prediction: {e}", 500
