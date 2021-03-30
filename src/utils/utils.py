import flask
import json


def to_json(data):
    return json.dumps(data) + "\n"


def resp_json(code, data):
    return flask.Response(
        status=code,
        mimetype="application/json",
        response=to_json(data)
    )
