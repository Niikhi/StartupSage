from flask import Flask
from .config import Config
from flask_cors import CORS
from .routes import main as main_blueprint

def create_app():
    # app = Flask(__name__, static_url_path='/static', static_folder='static')
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app)

    app.register_blueprint(main_blueprint)

    return app
