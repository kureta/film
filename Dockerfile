FROM nvcr.io/nvidia/tensorflow:21.03-tf2-py3

# set a directory for the app
WORKDIR /src

# install dependencies
RUN apt-get update
RUN apt-get --assume-yes install libsndfile-dev
RUN pip install --no-cache-dir ddsp==1.3.0 crepe==0.0.11

# define the port number the container should expose
EXPOSE 8888

# run the command
CMD ["jupyter", "notebook"]
