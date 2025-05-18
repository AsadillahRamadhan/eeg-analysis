from flask import Flask
from configs.db import Database, db, migrate
from app.routes.web import web
import os

def create_app():
    template_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
    static_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))

    app = Flask(__name__, template_folder=template_path, static_folder=static_path)
    app.secret_key = os.environ.get('SECRET_KEY', '12345')

    app.config.from_object(Database)
    db.init_app(app)
    migrate.init_app(app, db)

    app.register_blueprint(web)

    return app