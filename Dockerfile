FROM python:3.12
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p source models
COPY app.py .
COPY source/games.parquet source/
COPY models/ models/
CMD ["streamlit", "run", "app.py"]
