import json
import os

import boto3

boto_session = None
storage_client = None
docapi_table = None
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


def get_docapi_table():
    global docapi_table
    if docapi_table is not None:
        return docapi_table

    docapi_table = get_boto_session().resource(
        'dynamodb',
        endpoint_url=os.environ['DOCAPI_ENDPOINT'],
        region_name='ru-central1'
    ).Table('tasks')
    return docapi_table


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
    get_docapi_table().update_item(
        Key={'task_id': task_id},
        AttributeUpdates={
            'status': {'Value': 'queued', 'Action': 'PUT'},
            'zip_url': {'Value': zip_url, 'Action': 'PUT'},
        }
    )

    get_ymq_queue().send_message(MessageBody=json.dumps(
        {'task_id': task_id, "zip_url": zip_url}))
    return {
        'task_id': task_id
    }


def get_task_status(task_id):
    task = get_docapi_table().get_item(Key={
        "task_id": task_id
    })

    task_status = task['Item']['status']

    if task_status == 'completed':
        return {
            'status': task_status,
            'result_url': task['Item']['result_url']
        }
    return {'status': task_status}


def handle_api(event, context):
    action = event['action']
    if action == 'convert':
        return add_task_to_queue(event['task_id'], event['zip_url'])
    elif action == 'get_task_status':
        return get_task_status(event['task_id'])
    else:
        return {"error": "unknown action: " + action}
