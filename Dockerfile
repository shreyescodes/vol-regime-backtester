# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir build && python -m build --wheel --outdir /dist

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib
RUN useradd --create-home --uid 1000 quant
WORKDIR /home/quant
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl
COPY --chown=quant:quant configs ./configs
COPY --chown=quant:quant data/sample ./data/sample
USER quant
ENTRYPOINT ["quantvol"]
CMD ["run", "--config", "configs/spy.yaml", "--out", "reports"]
