#!/bin/bash

gunicorn main:app --bind 0.0.0.0:9999 --log-level DEBUG --reload