## level2-klue-nlp-01

```
|-- data_utils.py
|-- dataset
|   |-- dev
|   |-- dict_label_to_num.pkl
|   |-- dict_num_to_label.pkl
|   |-- test
|   `-- train
|-- metrics.py
|-- parameters
|   `-- roberta-large
|-- requirements.txt
|-- results
|-- settings.py
|-- train.py
|-- Dockerfile
|-- asset
`-- utils.py

```


### Docker setting
**1.clone this repository**
``` 
git clone https://github.com/boostcampaitech6/level2-klue-nlp-01.git
cd level2-klue-nlp-0
```

**2.build Dockerfile**
```
docker build --tag [filename]:1.0
```

**3.execute**

```
# Docker version 2.0 or later.
docker run -itd --runtime=nvidia --name dgl_tuto -p 8888:8888 -v C:\Users\Name\:/workspace [filename]:1.0 /bin/bash
```

```
# Docker-ce 19.03 or later
docker run -itd --gpus all --name boostcamp -p 8888:8888 -v C:\Users\Name\:/workspace [filename]:1.0 /bin/bash
```

**4.use jupyter notebook**
```
docker exec -it boostcamp bash

jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```
