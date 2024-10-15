import os

from flask import render_template, request, abort, Flask, jsonify, url_for
from flask.blueprints import Blueprint
from flask_socketio import SocketIO
from PIL.Image import Image
import pypdfium2 as pdfium

from application_model.deamons import FileChangeDaemon
from score_scan_ocr.defs import Instruments
from score_scan_ocr.ocr import parse_file, preprocess


bp = Blueprint("ocr", __name__, url_prefix="/score-ocr")
file_daemon = FileChangeDaemon()


@bp.route("/")
def index():
    return render_template("score_orc_result.html")


@bp.route("/files", methods=["post"])
def files():
    return render_template("snippets/filetree.html", files=file_daemon.files)


@bp.route("/set_root", methods=["post"])
def set_root():
    path = request.form.get("input")
    if path != file_daemon.path:
        try:
            file_daemon.start(path)
        except AttributeError:
            abort(400)
    return jsonify(success=True)


@bp.route("/load_data/<path>", methods=["post"])
def load_data(path: str):
    img_path = os.path.join(file_daemon.path, path)
    img = preprocess(img_path)
    img.save(f"flask_ui/static/temp/{path[:-4]}.jpg")
    doc = parse_file(img)
    print(doc)
    return render_template("snippets/prediction.html", img=path[:-4], instruments=Instruments.keys(), doc=doc)


def messaging(app: Flask, socketio: SocketIO) -> None:
    @socketio.event("files-changed")
    def files_changed(data):
        with app.test_request_context('/'):
            socketio.emit("files-changed")

    file_daemon._on_change = files_changed
