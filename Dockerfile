FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
COPY rp_handler.py /

# Start the container
CMD ["python3", "-u", "rp_handler.py"]