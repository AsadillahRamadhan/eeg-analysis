from flask import Blueprint
from app.controllers.auth_controller import AuthController
from app.controllers.dashboard_controller import DashboardController
from app.controllers.analyze_controller import AnalyzeController
from app.controllers.logs_controller import LogsController

web = Blueprint('web', __name__)

@web.get('/login')
def login_view():
    return AuthController.login_view()

@web.post('/login')
def login_process():
    return AuthController.login_process()

@web.get('/dashboard')
def dashboard_view():
    return DashboardController.index()

@web.get('/training')
def training_view():
    return AnalyzeController.training_view()

@web.post('/training')
def training_process():
    return AnalyzeController.training_process()

@web.get('/testing')
def analyze_view():
    return AnalyzeController.testing_view()

@web.post('/testing')
def testing_process():
    return AnalyzeController.testing_process()

@web.get('/logs')
def logs_view():
    return LogsController.index()

@web.post('/logs/delete/<id>')
def delete_logs(id):
    return LogsController.delete(id)

@web.post('/download')
def download():
    return AnalyzeController.download()