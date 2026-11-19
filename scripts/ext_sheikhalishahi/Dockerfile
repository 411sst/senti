# Runs Sheikhalishahi's eICU extraction (github.com/mostafaalishahi/eICU_Benchmark_updated
# at cd9b83d) with the package versions its readme pins. Extraction needs no Keras.
FROM python:3.7.17-slim-bullseye
RUN pip install --no-cache-dir numpy==1.15.0 scipy==1.2.0 scikit-learn==0.21.2 pandas==0.24.1
RUN useradd -m runner
USER runner
WORKDIR /work
