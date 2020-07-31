# Face id pycloud demo

Creating functional face id cloud service from less than 100 lines of code with PyCloud

## Preparation of python environment
Download pycloud from https://pycloud.ai#download

```virtualenv ~/.venv -p python3

. ~/.venv3/bin/activate

chmod a+x ./pycloud-install*.sh
./pycloud-install*.sh

pip install -r requirements.txt
```

## Deployment

### Docker
```
./pycloudcli deploy docker --file face_id_demo.py
```
### Microk8s
```
./pycloudcli deploy microk8s --file face_id_demo.py
```
### Google Cloud GKE
```
./pycloudcli deploy gke --file face_id_demo.py
```
### Amazon AWS EKS
```
./pycloudcli deploy eks --file face_id_demo.py
```

## Usage
Obtain service hosts and ports from `./pycloudcli` output.

### GRPC
Please check scripts: `detect.py` and `add_to_database.py` 

Sample usage:
```buildoutcfg
python add_to_database.py host 6779 ./obama.jpeg "Barack Obama"
python recognize.py host 6779 ./obama2.jpeg 
```
### HTTP
```
curl  host:5000 -F "image=@./chuck.jpeg" -F "name=Chuck Norris" -F "endpoint_id=register@face_id_demo"
curl  host:5000 -F "image=@./two_guys.jpeg" -F "endpoint_id=recognize@face_id_demo"
```

