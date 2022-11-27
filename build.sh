#!/bin/bash
# создает версию функции
#cat .env
source .env

echo "Собрать контейнер $SERERLESS_REGISTER_ID"
docker build . \
  -t cr.yandex/$SERERLESS_REGISTER_ID/ubuntu:hello

cat authorized_key.json | docker login \
--username json_key \
--password-stdin \
cr.yandex

docker push cr.yandex/$SERERLESS_REGISTER_ID/ubuntu:hello