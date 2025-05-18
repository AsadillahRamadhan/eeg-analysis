from flask import render_template, request, session, redirect, flash
from app.models.user import User

class AuthController:
    def login_view():
        return render_template('auth/login.html')
    
    def login_process():
        email, password = request.form['email'], request.form['password']
    
        user = User.query.filter_by(email=email).first()
        if(user and user.verify_password(password)):
            session['user_id'] = user.id
            session['username'] = user.username
            session['email'] = user.email
            session['role'] = user.role
            return redirect('/dashboard')
        else:
            flash("The credentials doesn't match our records!", "message")
            return redirect(request.referrer or '/')