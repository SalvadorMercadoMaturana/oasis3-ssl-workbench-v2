# Optional image with the workbench preinstalled (the RunPod flow uses the stock
# image + setup.sh instead). CUDA 12.4 + PyTorch 2.4, Python 3.11.
#   docker build -t oasis3-ssl-workbench .
# Data (data/roi3d, data/dementia, the split manifest) is mounted or copied at
# run time: it is OASIS3-derived and never baked into the image.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace/oasis3-ssl-workbench
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY setup.sh ./
RUN bash setup.sh
COPY . .
RUN python -m pip install --no-deps -e .

ENV PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
    OMP_NUM_THREADS=2

CMD ["bash", "bootstrap_runpod.sh"]
