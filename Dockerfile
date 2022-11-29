# syntax=docker/dockerfile:1

# start by pulling the python image
FROM python:3.9-slim-bullseye

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# switch working directory
WORKDIR /app

# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt --user

# copy every content from the local file to the image
COPY . /app

ENTRYPOINT [ "python" ]

CMD [ "./app/app.py" ]