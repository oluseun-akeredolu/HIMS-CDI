# ------------------------------------------------------------------
# STEP 1: Choose a base image
# ------------------------------------------------------------------
# python:3.11-slim is a lightweight Linux computer with Python 3.11
# already installed. "Slim" means it has no extra junk.
FROM python:3.11-slim

# ------------------------------------------------------------------
# STEP 2: Install system tools needed to compile numpy/scikit-learn
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------
# STEP 3: Set the working directory inside the container
# ------------------------------------------------------------------
# All commands after this run inside /app
WORKDIR /app

# ------------------------------------------------------------------
# STEP 4: Copy the frozen requirements and install them
# ------------------------------------------------------------------
COPY requirements-frozen.txt .
RUN pip install --no-cache-dir -r requirements-frozen.txt

# ------------------------------------------------------------------
# STEP 5: Copy the source code
# ------------------------------------------------------------------
COPY src/ /app/src/
COPY scripts/ /app/scripts/

# ------------------------------------------------------------------
# STEP 6: Copy the frozen model and its manifest
# ------------------------------------------------------------------
COPY artifacts/ /app/artifacts/

# ------------------------------------------------------------------
# STEP 7: Copy the deployment configuration (human-readable spec)
# ------------------------------------------------------------------
COPY deployment_config.json /app/

# ------------------------------------------------------------------
# STEP 8: Verify model integrity at build time
# ------------------------------------------------------------------
# If the SHA-256 does not match, the build fails immediately.
RUN python3 -c "import hashlib, json; m=json.load(open('/app/artifacts/frozen_model_real.manifest.json')); h=hashlib.sha256(open('/app/artifacts/frozen_model_real.joblib','rb').read()).hexdigest(); assert h == m['sha256'], 'Hash mismatch: ' + h + ' != ' + m['sha256']; print('Build-time integrity check passed')"

# ------------------------------------------------------------------
# STEP 9: Tell Python where to find our modules
# ------------------------------------------------------------------
ENV PYTHONPATH=/app/src:/app/scripts

# ------------------------------------------------------------------
# STEP 10: Define the command that runs when the container starts
# ------------------------------------------------------------------
CMD ["python3", "-m", "csra.streaming_engine"]