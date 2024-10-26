#!/bin/sh

VERSION_STREAM=$(git rev-parse --short HEAD) #or add your version here

REPOSITORY=docker.io/lamdaleon #public repo


# docker build -t $REPOSITORY/nginx-ingress:$VERSION_STREAM -f ./Dockerfile_nginx ./src
# docker push $REPOSITORY/nginx-ingress:$VERSION_STREAM
#
# docker build -t $REPOSITORY/frontend:$VERSION_STREAM -f ./src/frontend/Dockerfile ./src/frontend
# docker push $REPOSITORY/frontend:$VERSION_STREAM
#
# docker build -t $REPOSITORY/localapi:$VERSION_STREAM -f ./src/localapi/Dockerfile ./src/localapi/
# docker push $REPOSITORY/localapi:$VERSION_STREAM

docker build -t $REPOSITORY/targetx-api-server:$VERSION_STREAM -f ./Dockerfile_api_server ./
docker push $REPOSITORY/targetx-api-server:$VERSION_STREAM


# docker build -t $REPOSITORY/trainer:$VERSION_STREAM -f ./Dockerfile_trainer ./src
# docker push $REPOSITORY/trainer:$VERSION_STREAM
