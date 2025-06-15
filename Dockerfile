FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

COPY model.pkl .
COPY inference.py .

EXPOSE 8080
ENTRYPOINT ["python", "inference.py"]

