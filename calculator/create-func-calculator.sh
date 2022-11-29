#!/bin/bash
# создает версию функции
#cat .env
source ../.env

echo "Удалить $CALCULATOR_FUNCTION_NAME"
yc serverless function delete --name=$CALCULATOR_FUNCTION_NAME

echo "Создать $CALCULATOR_FUNCTION_NAME"
yc serverless function create --name=$CALCULATOR_FUNCTION_NAME
# echo "сделать функцию публичной"
# yc serverless function allow-unauthenticated-invoke $AUTH_FUNCTION_NAME
