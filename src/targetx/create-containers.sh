#!/bin/sh

VERSION_STREAM=$(git rev-parse --short HEAD)

REPOSITORY=docker.io/lamdaleon
#REPOSITORY=gcr.io/privacyanalyzer-407121

# docker build -t $REPOSITORY/train_lstm_image:$VERSION_STREAM -f ./Dockerfile_train_lstm ./
# docker push $REPOSITORY/train_lstm_image:$VERSION_STREAM

# docker build -t $REPOSITORY/my_nginx:$VERSION_STREAM -f ./Dockerfile_nginx ./
# docker push $REPOSITORY/my_nginx:$VERSION_STREAM

docker build -t $REPOSITORY/frontend:$VERSION_STREAM -f ../pvapp/frontend/Dockerfile /Users/leonidas/OneDrive/TargetX26July/project/code/targetx/FINAL/pvapp/frontend
docker push $REPOSITORY/frontend:$VERSION_STREAM

# docker build -t $REPOSITORY/localapi:$VERSION_STREAM -f ../pvapp/localapi/Dockerfile /Users/leonidas/OneDrive/TargetX26July/project/code/targetx/FINAL/pvapp/localapi/
# docker push $REPOSITORY/localapi:$VERSION_STREAM

# docker build -t $REPOSITORY/targetx-api-server:$VERSION_STREAM -f ./Dockerfile_api_server ./
# docker push $REPOSITORY/targetx-api-server:$VERSION_STREAM
