#!/bin/bash
rm -R ./build
rm -R ./dist
mkdir build
source ../.env

# сделать архив
cd ./app || exit
rm ../build/func.zip
7z a -tzip ../build/func.zip .
cd ..

echo "yc function version create $TEST_FUNCTION_NAME "

yc serverless function version create \
  --function-name=$TEST_FUNCTION_NAME \
  --runtime python39 \
  --entrypoint test.handler \
  --memory 1024m \
  --execution-timeout 600s \
  --source-path ./build/func.zip \
  --service-account-id=$TEST_SERVICE_ACCOUNT_ID \
  --folder-id $TEST_FOLDER_ID \
  --environment TEST_SA_ACCESS_KEY_ID=$TEST_SA_ACCESS_KEY_ID,TEST_SA_ACCESS_SECRET_KEY=$TEST_SA_ACCESS_SECRET_KEY,TEST_FUNCTION_NAME=$TEST_FUNCTION_NAME,TEST_FOLDER_ID=$TEST_FOLDER_ID,TEST_ENDPOINT=$TEST_ENDPOINT,TEST_DATABASE=$TEST_DATABASE,TEST_DOCAPI_ENDPOINT=$TEST_DOCAPI_ENDPOINT
