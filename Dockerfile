FROM python:3.12
WORKDIR /app
COPY requirements.txt app.py .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir source models
COPY "source/games.parquet" source
COPY "models/*" models
CMD [ "streamlit", "run", "app.py" ]