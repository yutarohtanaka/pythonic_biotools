FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        zlib1g-dev \
        libbz2-dev \
        liblzma-dev \
        libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/pythonic_biotools

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
        "numpy>=1.24" \
        "scipy>=1.10" \
        "pandas>=2.0" \
        "scikit-learn>=1.3" \
        "anndata>=0.10" \
        "pysam>=0.22" \
        "requests>=2.31" \
        "matplotlib>=3.7" \
        "pytest>=7.4"

COPY . /opt/pythonic_biotools

RUN python -m pip install --no-deps -e . && \
    python -m pythonic_biotools.cli install all --no-deps && \
    pbt check-imports --bundle all

ENTRYPOINT ["pbt"]
CMD ["--help"]
