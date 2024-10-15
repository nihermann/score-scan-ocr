from flask import render_template, request, abort
from flask.blueprints import Blueprint

bp = Blueprint("split", __name__, url_prefix="/split")


@bp.route("/")
def index():
    return render_template("filesplitter.html")
