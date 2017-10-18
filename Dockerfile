# Visit https://hub.docker.com/r/tensorflow/tensorflow/tags/
# to see available version.

FROM gcr.io/tensorflow/tensorflow:0.12.0-gpu

RUN apt-get update && apt-get install -y \
    git libhdf5-dev

# Set the working directory to /eai, and set the HOME environment
# variable too.
WORKDIR /eai/project
ENV HOME /eai/project


# Add Python dependencies files and install them
COPY requirements.txt /eai/
RUN pip install -r /eai/requirements.txt
