# FastAPI - Naive Bayes Classifier
[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Implementation Naive Bayes Classifier algorithm for classifying book titles.
## Installation
Install the dependencies and devDependencies and start the server.

```sh
pip3.9 install virtualenv
python -m venv env 
.\env\Scripts\activate
```
Installing Library:
```sh
pip install fastapi
pip install uvicorn
pip install PySastrawi
```
Check requirements installed:
```sh
pip freeze
```
Read requirements on requirements.txt
```sh
pip install -r requirements.txt
```
Starting server:
```sh
uvicorn main:app
```
Starting server with hot reload:
```sh
uvicorn main:app --reload
```