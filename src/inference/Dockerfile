FROM python:3.11.6

# Changing the path to the working directory in the container
WORKDIR /app

# Copy the code itself
COPY src/inference /app/src/inference

# Copy data folder to Docker
COPY ../data /app/data

# Copy models folder to Docker
COPY ../../outputs/models /app/outputs/models

#Copy utils.py
COPY ../../utils.py /app

# Install any necessary packages listed in requirements.txt
COPY ../../requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Trains the model after running instance of this image
CMD ["python3","src/inference/inference_run.py"]