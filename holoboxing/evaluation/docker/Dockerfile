# Both bases are public. Release builds pass their verified digest references.
ARG PYTHON_BASE=python:3.11.14-slim-bookworm
ARG NODE_BASE=node:22.14.0-bookworm-slim
FROM ${NODE_BASE} AS viewer
WORKDIR /viewer
COPY viewer/package.json viewer/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --ignore-scripts
COPY viewer/build.mjs ./
COPY viewer/src ./src
COPY viewer/notices ./notices
RUN node build.mjs

FROM ${PYTHON_BASE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app HOLOBOXING_ASSETS=/opt/holoboxing/assets \
    HOLOBOXING_LOWER=/opt/holoboxing/lower \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    XDG_CACHE_HOME=/tmp/holoboxing-cache CUDA_CACHE_PATH=/tmp/holoboxing-cache/cuda \
    WARP_CACHE_PATH=/tmp/holoboxing-cache/warp TORCH_HOME=/tmp/holoboxing-cache/torch
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libegl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Canonical Torch observation code is exported and compared on the CPU here.
# This stage is not included in the participant image.
FROM base AS observation-build
COPY build-requirements.lock /tmp/build-requirements.lock
RUN --mount=type=cache,target=/tmp/holoboxing-cache/pip \
    pip install --require-hashes --only-binary=:all: -r /tmp/build-requirements.lock
COPY app/ /app/
COPY assets/ /opt/holoboxing/assets/
COPY lower/ /opt/holoboxing/lower/
RUN python -m holoboxing.evaluation.robot_models --assets /opt/holoboxing/assets
RUN python -m holoboxing.evaluation.runtime.observation_export \
      --assets /opt/holoboxing/assets --lower /opt/holoboxing/lower \
      --output /app/holoboxing/evaluation/runtime/compiled

FROM base AS runtime
COPY requirements.lock /tmp/requirements.lock
# pip follows XDG_CACHE_HOME above; keep wheel downloads out of image layers.
RUN --mount=type=cache,target=/tmp/holoboxing-cache/pip \
    pip install --require-hashes --only-binary=:all: -r /tmp/requirements.lock
COPY app/ /app/
COPY --from=observation-build /app/holoboxing/evaluation/runtime/compiled /app/holoboxing/evaluation/runtime/compiled
COPY --from=viewer /viewer/dist /app/holoboxing/evaluation/viewer/dist
COPY assets/ /opt/holoboxing/assets/
COPY lower/ /opt/holoboxing/lower/
COPY build-inputs.json /opt/holoboxing/build-inputs.json
RUN python -m holoboxing assets \
    && python -m holoboxing.evaluation.robot_models --assets /opt/holoboxing/assets \
    && python -c 'from holoboxing.evaluation.models import verify_bundle; verify_bundle("/opt/holoboxing/lower")' \
    && python -c 'from holoboxing.evaluation.runtime.compiled_artifacts import manifest; manifest()' \
    && python -m pip check \
    && dpkg-query -W > /opt/holoboxing/system-packages.txt \
    && python -m pip freeze > /opt/holoboxing/python-packages.txt
ARG CODE_REVISION=unknown
ARG INPUTS_SHA256=unknown
LABEL org.opencontainers.image.title="HoloBoxing GPU evaluation" \
      org.opencontainers.image.version="v10" \
      org.opencontainers.image.source="https://github.com/HorizonRobotics/HoloMotion" \
      org.opencontainers.image.revision="${CODE_REVISION}" \
      org.holomotion.holoboxing.inputs.sha256="${INPUTS_SHA256}" \
      org.holomotion.holoboxing.robot-contract="urdf-gloves-v1" \
      org.holomotion.holoboxing.collision-precision="epa-plane-fp64-v1"
EXPOSE 8080
ENTRYPOINT ["python", "-m", "holoboxing"]
CMD ["--help"]
