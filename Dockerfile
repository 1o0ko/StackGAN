# Visit https://hub.docker.com/r/tensorflow/tensorflow/tags/
# to see available version.

FROM gcr.io/tensorflow/tensorflow:0.12.0-gpu

RUN apt-get update && apt-get install -y \
    git libhdf5-dev

# why do we need it?
RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory to /eai, and set the HOME environment
# variable too.
WORKDIR /eai/project
ENV HOME /eai/project


# Add Python dependencies files and install them
COPY requirements.txt /eai/
RUN pip install -r /eai/requirements.txt


ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD ["jupyter", "notebook"]
