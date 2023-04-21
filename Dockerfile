# Use an official Python runtime as a parent image
FROM python:3.9.16-bullseye

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the entire project directory into the container
COPY . /app

# Expose the ports that the apps will run on
EXPOSE 8000 8501

# Start the FastAPI and Streamlit apps in the background
CMD ["sh", "-c", "uvicorn src.api.api:app --host 0.0.0.0 --port 8000 & streamlit run src/streamlit/streamlit.py"]