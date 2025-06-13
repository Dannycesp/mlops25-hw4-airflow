# Use the base image provided in the homework
FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the generated requirements.txt file into the container
COPY requirements.txt .

# Install the Python dependencies directly using pip
# This is a more robust and standard method for Docker builds.
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application script into the container
COPY starter.py .

# Define the command that will be executed when the container starts.
ENTRYPOINT ["python", "starter.py"]
