import json
import os

import ydb
import boto3

import yandexcloud

from yandex.cloud.serverless.apigateway.websocket.v1.connection_service_pb2 import SendToConnectionRequest
from yandex.cloud.serverless.apigateway.websocket.v1.connection_service_pb2_grpc import ConnectionServiceStub

sa_key_file = open('authorized_key.json')
data = json.load(sa_key_file)

script_dir = os.path.dirname(__file__)
rel_path = "authorized_key.json"
abs_file_path = os.path.join(script_dir, rel_path)

with open(abs_file_path) as json_file:
    sa_key = json.load(json_file)

sdk = yandexcloud.SDK(service_account_key=sa_key)

ws_client = sdk.client(ConnectionServiceStub)

boto_session = None
storage_client = None
db_pool = None
ymq_queue = None


def get_boto_session():
    global boto_session
    if boto_session is not None:
        return boto_session

    # extract values from secret
    access_key = os.environ['TEST_SA_ACCESS_KEY_ID']
    secret_key = os.environ['TEST_SA_ACCESS_SECRET_KEY']

    if access_key is None or secret_key is None:
        raise Exception("secrets required")
    print("Key id: " + access_key)

    # initialize boto session
    boto_session = boto3.session.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    return boto_session


def get_db_pool():
    global db_pool
    if db_pool is not None:
        return db_pool

    # Create driver in global space.
    driver = ydb.Driver(
        endpoint=os.environ['TEST_ENDPOINT'], database=os.environ['TEST_DATABASE'])
    # Wait for the driver to become active for requests.
    driver.wait(fail_fast=True, timeout=5)
    # Create the session pool instance to manage YDB sessions.
    db_pool = ydb.SessionPool(driver)

    return db_pool


# Converter handler

def get_connection_id(task_id):
    def callee(session):
        prepared_query = session.prepare(
            f"""
                SELECT u.ws_connection_id
                FROM users u
                INNER JOIN requests r ON r.user_id=u.id
                WHERE r.id="{task_id}"
                LIMIT 1
            """)
        session.transaction().execute(prepared_query, commit_tx=True)

    result = get_db_pool().retry_operation_sync(callee)

    print(f"result: {result}")

    return result['ResultSets'][0]


def handler(event, context):
    task_json = json.loads(event['body'])
    task_id = task_json['task_id']
    conn_id = task_json['conn_id']

    print(f'Processing task. task_id:{task_id}')

    connection_id = conn_id
    msg = json.dumps({"test": 1}).encode('utf-8')

    SendToConnectionRequest(
        connection_id=connection_id,
        data=msg,
        type=2
    )

    return "OK"
