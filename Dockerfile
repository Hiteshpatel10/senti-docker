FROM python:3.8-slim-buster

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc virtualenv && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN virtualenv -p python venv
ENV PATH="/venv/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the application code and the requirements file to the container
COPY . /app

# Install the dependencies and download NLTK data
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import nltk; nltk.download('all', download_dir='/nltk_data')"

# Set NLTK data path environment variable
ENV NLTK_DATA=/nltk_data

# Expose the Flask port
EXPOSE 5000

# Run the Flask app and the delete_old_files.py daemon
CMD ["python", "-u", "app.py", "&", "python", "-u", "delete_old_files.py"]
