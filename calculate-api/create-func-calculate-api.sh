#!/bin/bash
# создает версию функции
#cat .env
source ../.env

echo "Удалить $CALCULATE_API_FUNCTION_NAME"
yc serverless function delete --name=$CALCULATE_API_FUNCTION_NAME

echo "Создать $CALCULATE_API_FUNCTION_NAME"
yc serverless function create --name=$CALCULATE_API_FUNCTION_NAME
# echo "сделать функцию публичной"
# yc serverless function allow-unauthenticated-invoke $AUTH_FUNCTION_NAME
