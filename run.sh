# chmod +x ./run.sh

EXPOSE_PORT=8123 \
    CONTAINER_NAME=ids_uit-$EXPOSE_PORT \
    CONFIG_SET=prod \
    CURRENT_UID=$(id -u):$(id -g) \
    docker-compose down --remove-orphans

EXPOSE_PORT=8123 \
    CONTAINER_NAME=ids_uit-$EXPOSE_PORT \
    CONFIG_SET=prod \
    CURRENT_UID=$(id -u):$(id -g) \
    IMAGE_NAME=detect-server:1.0 \
    docker-compose build --force-rm

EXPOSE_PORT=8123 \
    CONTAINER_NAME=ids_uit-$EXPOSE_PORT \
    CONFIG_SET=prod \
    CURRENT_UID=$(id -u):$(id -g) \
    IMAGE_NAME=detect-server:1.0 \
    docker-compose up -d --remove-orphans detect-server
