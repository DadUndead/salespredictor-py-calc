#!/bin/bash
rm -R ./build
rm -R ./dist
mkdir build
source ../.env

# сделать архив
cd ./app
rm ../build/func.zip
7z a -tzip ../build/func.zip .
cd ..

echo "yc function version create $CALCULATOR_FUNCTION_NAME "

yc serverless function version create \
  --function-name=$CALCULATOR_FUNCTION_NAME \
  --runtime python39 \
  --entrypoint calculator.handle_process_event \
  --memory 256m \
  --execution-timeout 30s \
  --source-path ./build/func.zip \
  --service-account-id=$SERVICE_ACCOUNT_ID \
  --folder-id $FOLDER_ID \
  --environment SA_ACCESS_KEY_ID=$SA_ACCESS_KEY_ID,SA_ACCESS_SECRET_KEY=$SA_ACCESS_SECRET_KEY,YMQ_QUEUE_URL=$YMQ_QUEUE_URL,BUCKET_NAME=$BUCKET_NAME,DOCAPI_ENDPOINT=$DOCAPI_ENDPOINT
