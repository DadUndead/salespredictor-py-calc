#!/bin/bash
# создает версию serverless-container
#cat .env
source .env

echo "Собрать контейнер $SERERLESS_REGISTER_ID"
docker build . \
  --platform=linux/amd64 \
  --pull \
  --rm \
  -f "Dockerfile" \
  -t cr.yandex/$SERERLESS_REGISTER_ID/ubuntu:0.0.1 \

cat authorized_key.json | docker login \
--username json_key \
--password-stdin \
cr.yandex

docker push cr.yandex/$SERERLESS_REGISTER_ID/ubuntu:0.0.1

yc serverless container revision deploy \
  --folder-id ${FOLDER_ID} \
  --container-id  ${CONTAINER_ID} \
  --async \
  --memory 512MB \
  --cores 1 \
  --execution-timeout 30s \
  --core-fraction 100 \
  --service-account-id $SERVICE_ACCOUNT_ID \
  --image cr.yandex/$SERERLESS_REGISTER_ID/ubuntu:0.0.1 \
  --environment SA_ACCESS_KEY_ID=$SA_ACCESS_KEY_ID,SA_ACCESS_SECRET_KEY=$SA_ACCESS_SECRET_KEY,BUCKET_NAME=$BUCKET_NAME,BUCKET_REGION=$BUCKET_REGION,TOKEN_KEY=$TOKEN_KEY

