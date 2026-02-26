FROM python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Runtime libs for cv2 / PIL / scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Build from inside "CFM HES":
#   cd "CFM HES"
#   docker build -t cfm-hes-demo .
COPY . /app/cfm_hes/

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/cfm_hes/requirements.txt

WORKDIR /app/cfm_hes
ENV PYTHONPATH=/app:/app/cfm_hes

EXPOSE 8501

CMD ["streamlit", "run", "Demo/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.fileWatcherType=none"]
