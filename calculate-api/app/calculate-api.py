import json
import os
import ydb

import boto3

boto_session = None
storage_client = None
db_pool = None
ymq_queue = None


def get_boto_session():
    global boto_session
    if boto_session is not None:
        return boto_session

    # extract values from secret
    access_key = os.environ['SA_ACCESS_KEY_ID']
    secret_key = os.environ['SA_ACCESS_SECRET_KEY']

    if access_key is None or secret_key is None:
        raise Exception("secrets required")
    print("Key id: " + access_key)

    # initialize boto session
    boto_session = boto3.session.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    return boto_session


def get_ymq_queue():
    global ymq_queue
    if ymq_queue is not None:
        return ymq_queue

    ymq_queue = get_boto_session().resource(
        service_name='sqs',
        endpoint_url='https://message-queue.api.cloud.yandex.net',
        region_name='ru-central1'
    ).Queue(os.environ['YMQ_QUEUE_URL'])
    return ymq_queue


def get_db_pool():
    global db_pool
    if db_pool is not None:
        return db_pool

    # Create driver in global space.
    driver = ydb.Driver(
        endpoint=os.environ['ENDPOINT'], database=os.environ['DATABASE'])
    # Wait for the driver to become active for requests.
    driver.wait(fail_fast=True, timeout=5)
    # Create the session pool instance to manage YDB sessions.
    db_pool = ydb.SessionPool(driver)

    return db_pool


def get_storage_client():
    global storage_client
    if storage_client is not None:
        return storage_client

    storage_client = get_boto_session().client(
        service_name='s3',
        endpoint_url='https://storage.yandexcloud.net',
        region_name='ru-central1'
    )
    return storage_client

# API handler


def add_task_to_queue(task_id, zip_url):
    def callee(session):
        prepared_query = session.prepare(
            f"""
                UPDATE requests
                SET status = 'queued', zip_url = '{zip_url}'
                WHERE id = "{task_id}"
            """)

        session.transaction().execute(prepared_query, commit_tx=True)

    get_db_pool().retry_operation_sync(callee)

    get_ymq_queue().send_message(MessageBody=json.dumps(
        {'task_id': task_id, "zip_url": zip_url}))

    return {
        'task_id': task_id
    }


def get_task_status(task_id):

    def callee(session):
        prepared_query = session.prepare(
            f'SELECT id, status, result_url FROM requests WHERE id = "{task_id}"')
        result_sets = session.transaction().execute(
            prepared_query,
            commit_tx=True,
        )

        return result_sets[0].rows[0]

    task = get_db_pool().retry_operation_sync(callee)

    print(f'task:{task}')

    if task['status'] == 'completed':
        return {
            'status': task['status'],
            'result_url': task['result_url']
        }

    return {'status': task['status']}


def handle_api(event, context):
    action = event['action']
    if action == 'convert':
        return add_task_to_queue(event['task_id'], event['zip_url'])
    elif action == 'get_task_status':
        return get_task_status(event['task_id'])
    else:
        return {"error": "unknown action: " + action}
