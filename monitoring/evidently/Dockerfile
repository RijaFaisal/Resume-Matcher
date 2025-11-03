FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    evidently==0.4.33 \
    pandas==2.0.3

# Create workspace and reports directories
RUN mkdir -p /app/workspace /app/reports

EXPOSE 7000

CMD ["evidently", "ui", "--workspace", "/app/workspace", "--port", "7000", "--host", "0.0.0.0"]