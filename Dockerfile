FROM monai/monai:latest

# Install additional packages
RUN pip install antspyx

COPY . /learnfmri

# Add the path to the PYTHONPATH
ENV PYTHONPATH="/learnfmri:${PYTHONPATH}"
