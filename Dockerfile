FROM python:3.12
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
run mkdir source models
COPY "source/games.parquet" "source/games.parquet"
COPY "models/*" models
CMD [ "streamlit", "run", "app.py" ]