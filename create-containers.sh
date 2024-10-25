#!/bin/sh

VERSION_STREAM=$(git rev-parse --short HEAD) #or add your version here

REPOSITORY=docker.io/lamdaleon


docker build -t $REPOSITORY/nginx-ingress:$VERSION_STREAM -f ./Dockerfile_nginx ./src
docker push $REPOSITORY/nginx-ingress:$VERSION_STREAM

# docker build -t $REPOSITORY/frontend:$VERSION_STREAM -f ../pvapp/frontend/Dockerfile /Users/leonidas/OneDrive/TargetX26July/project/code/targetx/FINAL/pvapp/frontend
# docker push $REPOSITORY/frontend:$VERSION_STREAM

# docker build -t $REPOSITORY/localapi:$VERSION_STREAM -f ../pvapp/localapi/Dockerfile /Users/leonidas/OneDrive/TargetX26July/project/code/targetx/FINAL/pvapp/localapi/
# docker push $REPOSITORY/localapi:$VERSION_STREAM

# docker build -t $REPOSITORY/targetx-api-server:$VERSION_STREAM -f ./Dockerfile_api_server ./src
# docker push $REPOSITORY/targetx-api-server:$VERSION_STREAM

# To train your model, you need to add the raw data in folder powerData. You will need to modify the target-trainer.py according to your data format
# docker build -t $REPOSITORY/trainer:$VERSION_STREAM -f ./Dockerfile_trainer ./src
# docker push $REPOSITORY/trainer:$VERSION_STREAM
