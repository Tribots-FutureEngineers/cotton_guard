# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /code

# Copy requirements file first to leverage Docker cache
COPY requirements.txt /code/requirements.txt

# Install dependencies (uses CPU version of PyTorch to make build fast and keep memory footprint low)
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the backend code and weights
COPY ./backend /code/backend

# Copy the frontend built assets (enables hosting the entire app directly on Hugging Face as well)
COPY ./frontend_built /code/frontend_built

# Set environment variable for port (Hugging Face default is 7860)
ENV PORT=7860
EXPOSE 7860

# Run uvicorn on port 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
