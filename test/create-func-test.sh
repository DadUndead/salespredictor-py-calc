#!/bin/bash
# создает версию функции
#cat .env
source ../.env

echo "Удалить $TEST_FUNCTION_NAME"
yc --cloud-id b1gq9ovhi9k06hmvgk88 serverless  function delete --name=$TEST_FUNCTION_NAME

echo "Создать $TEST_FUNCTION_NAME"
yc --cloud-id b1gq9ovhi9k06hmvgk88 serverless function create --name=$TEST_FUNCTION_NAME
# echo "сделать функцию публичной"
# yc serverless function allow-unauthenticated-invoke $AUTH_FUNCTION_NAME
