#!/bin/sh

gunicorn -b 0.0.0.0:5000 app:app &

sleep 10

$HOME/opt/ngrok/ngrok http 0.0.0.0:5000

