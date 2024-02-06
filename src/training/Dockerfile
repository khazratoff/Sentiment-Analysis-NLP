FROM python:3.11.6

#Adding MODEL_NAME as build argument
ARG MODEL_NAME="final_model.joblib"

# Setting the build arguments as environment variable to access through training.py script
ENV MODEL_NAME=${MODEL_NAME}

# Changing the path to the working directory in the container
WORKDIR /app



# Copy the code itself
COPY src/training /app/src/training

# Copy the data_loader 
COPY src/data_loader.py /app/src

#Copy utils.py
COPY ../../utils.py /app

# Install any necessary packages listed in requirements.txt
COPY ../../requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

#Loading data
RUN python3 src/data_loader.py

# Trains the model after running instance of this image
CMD ["python3","src/training/train.py"]

