This file contains simple instructions to setup the ollama code
conda create --name llama python=3.12
conda activate llama
pip install chromadb
pip install langchain-ollama
pip install pymupdf
pip install pydantic fastapi
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

