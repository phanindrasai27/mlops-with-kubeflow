# Use the official Python 3.8 image as the base image
FROM python:3.8

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install the required Python packages
RUN pip install -r requirements.txt

# Copy the source code files to the working directory
COPY . .

# Define the command to run when the container starts
CMD ["python", "app.py"]