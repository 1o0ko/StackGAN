# Visit https://hub.docker.com/r/tensorflow/tensorflow/tags/
# to see available version.

FROM gcr.io/tensorflow/tensorflow:0.12-gpu

# Set the working directory to /eai, and set the HOME environment
# variable too.
WORKDIR /eai/project
ENV HOME /eai/project


# Add Python dependencies files and install them
COPY requirements.txt /eai/
RUN pip install -r requirements.txt
