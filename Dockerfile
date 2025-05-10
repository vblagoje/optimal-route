# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install uv
RUN pip install uv

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt using uv
RUN uv pip install --system --no-cache -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Environment variables (to be provided at runtime)
ENV GOOGLE_API_KEY=""
ENV MCP_TRANSPORT="sse"

# Run the application
CMD python router.py --port 8080 --transport ${MCP_TRANSPORT} 