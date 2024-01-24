FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN pip install pandas==1.3.2 numpy==1.20 seaborn==0.13.1

RUN pip install matplotlib==3.7.4 transformers==4.10.0 scikit-learn==0.24.1

WORKDIR /workspace 

CMD ["bash"]
