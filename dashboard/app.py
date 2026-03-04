"""
LLM Benchmark Web Dashboard — Flask Application.
"""

from flask import Flask, send_from_directory
from backend.api import api

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)

# Register API blueprint
app.register_blueprint(api)


@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
