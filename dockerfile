
FROM python:3.9


WORKDIR /code


COPY ./app /code/app/


COPY ./requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


CMD ["fastapi", "run", "app/main.py", "--port", "80"]