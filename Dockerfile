FROM nvcr.io/nvidia/tensorflow:21.11-tf2-py3

# set a directory for the app
WORKDIR /src
COPY requirements.txt /tmp/requirements.txt

# install dependencies
RUN apt-get update
RUN apt-get --assume-yes install libsndfile-dev
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# define the port number the container should expose
EXPOSE 8888

# run the command
CMD ["jupyter", "notebook"]
