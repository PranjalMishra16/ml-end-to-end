
from flask import Flask, request, jsonify
import pickle
import numpy as np
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/predict", methods=["POST"])
def predict():
    input_json = request.get_json()
    x_val = input_json["x"]
    x_arr = np.array([[x_val]])
    prediction = model.predict(x_arr)
    return jsonify({"prediction": prediction[0]})

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ping":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        raw_data = self.rfile.read(length)
        data = json.loads(raw_data)

    # Assuming the input is in the v2 format:
        input_data = data.get("inputs", [])[0]["data"]
        pred = model.predict([input_data])[0]
        pred = float(pred)

        self.send_response(200)
        self.send_header("Content-Type", "application/vnd.sagemaker.inference.v2+json")
        self.end_headers()

        response = {
            "outputs": [{
                "name": "prediction",
                "shape": [1],
                "datatype": "FP32",
                "data": [pred]
            }]
        }
        self.wfile.write(json.dumps(response).encode("utf-8"))



def serve():
    server = HTTPServer(("0.0.0.0", 8080), Handler)
    print("Starting inference server on port 8080...")
    server.serve_forever()

if __name__ == "__main__":

    if len(sys.argv) == 1 or sys.argv[1] == "serve":
        serve()
    else:
        print(f"Unknown command: {sys.argv[1]}")
        sys.exit(1)