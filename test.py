import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {"mp4"}

app = Flask(__name__)
app.config['static'] = UPLOAD_FOLDER