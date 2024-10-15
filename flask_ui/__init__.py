import mimetypes
from typing import Tuple

from flask_socketio import SocketIO, emit

mimetypes.add_type('application/javascript', '.js')
from flask import Flask, render_template, abort


def create_app(test_config=None) -> Flask:
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    socketio = SocketIO(app)

    @app.route('/')
    @app.route('/hello')
    @app.route('/hello/<name>')
    def hello(name=None):
        return render_template('hello.html', name=name)

    from . import score_ocr
    app.register_blueprint(score_ocr.bp)
    score_ocr.messaging(app, socketio)

    from . import split
    app.register_blueprint(split.bp)

    @socketio.on('connect')
    def handle_connect():
        print("online")
        # Send an update to the client
        emit('update', {'data': 'Hello, client!'})

    return app


if __name__ == '__main__':
    app = create_app()


