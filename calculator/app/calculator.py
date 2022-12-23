import json
import os
from zipfile import ZipInfo, ZipFile

import ydb
import boto3

import time

from ydb.convert import ResultSet

import data_parse
import calculate as calc
import io
import pandas as pd
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


# Converter handler


def unzip_archive(zip_url, dest):
    client = get_storage_client()
    bucket = os.environ['BUCKET_NAME']
    path_to_zip_file = '/tmp/request.zip'
    print(f'Downloading {zip_url} to {path_to_zip_file}.')
    client.download_file(bucket, zip_url, path_to_zip_file)
    print(f'Unzipping {path_to_zip_file} to {dest}.')
    with ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(dest)


def upload_and_presign(file_path, object_name):
    client = get_storage_client()
    bucket = os.environ['BUCKET_NAME']
    client.upload_file(file_path, bucket, object_name)
    return client.generate_presigned_url('get_object', Params={'Bucket': bucket, 'Key': object_name}, ExpiresIn=604800)


def df_to_excel(df):
    # Creating output and writer (pandas excel writer)
    out = io.BytesIO()
    writer = pd.ExcelWriter(out, engine='xlsxwriter')

    # Export data frame to excel
    df.to_excel(excel_writer=writer, index=True, sheet_name='Sheet1')
    writer.save()
    writer.close()

    return out.getvalue()


def start_processing_task(task_id):
    def callee(session):
        prepared_query = session.prepare(
            f"""
                UPDATE requests
                SET status = 'processing'
                WHERE id = "{task_id}"
            """)
        session.transaction().execute(prepared_query, commit_tx=True)

    get_db_pool().retry_operation_sync(callee)


def update_task_status(task_id, status):
    def callee(session):
        prepared_query = session.prepare(
            f"""
                UPDATE requests
                SET status = "{status}"
                WHERE id = "{task_id}"
            """)
        session.transaction().execute(prepared_query, commit_tx=True)

    get_db_pool().retry_operation_sync(callee)


def complete_task(task_id, result_download_url):
    def callee(session):
        current_datetime = round(time.time() * 1000)
        prepared_query = session.prepare(
            f"""
                UPDATE requests
                SET status = 'complete', finished_at = DateTime::FromMilliseconds({current_datetime}), result_url = '{result_download_url}'
                WHERE id = "{task_id}"
            """)
        session.transaction().execute(prepared_query, commit_tx=True)

    get_db_pool().retry_operation_sync(callee)


def get_connection_id(task_id):
    def callee(session):
        prepared_query = session.prepare(
            f"""
                SELECT u.ws_connection_id as connection_id
                FROM users u
                INNER JOIN requests r ON r.user_id=u.id
                WHERE r.id="{task_id}"
                LIMIT 1
            """)
        return session.transaction().execute(prepared_query, commit_tx=True)

    result_set: ResultSet = get_db_pool().retry_operation_sync(callee)[0]

    print(f"result: {result_set.rows[0]}")

    return result_set.rows[0]["connection_id"]


def send_status_ws_message(task_id, status):
    try:
        connection_id = get_connection_id(task_id)
        msg = json.dumps({
            "type": "status-update",
            "payload": {
                "type": 'requestStatusChange',
                "requestId": task_id,
                "status": status
            }
        }).encode('utf-8')

        print(f'connection_id:' + connection_id)
        print(f'Sending ws message:' + str(msg))

        ws_request = SendToConnectionRequest(
            connection_id=connection_id,
            data=msg,
            type=2,
        )

        ws_client.Send(ws_request)

    except Exception as e:
        print(f'Error on sending ws message. ' + str(e))


def handle_process_event(event, context):
    for message in event['messages']:
        task_json = json.loads(message['details']['message']['body'])
        task_id = task_json['task_id']
        zip_url = task_json['zip_url']

        # update request status in db
        # send ws message {type: 'status-update', requestId, status: 'processing'}

        print(f'Processing task. task_id:{task_id} zip_url:{zip_url}')

        # Update task status
        start_processing_task(task_id)
        send_status_ws_message(task_id, 'processing')

        print('Unzipping request archive...')
        unzip_archive(zip_url, '/tmp')

        # Calculate result

        try:
            data = data_parse('/tmp/data.xlsx', '/tmp/data_input.xlsx')
        except Exception as e:
            print(f'Error (data_parse): Failed to parse data files. ' + str(e))
            update_task_status(task_id, 'error')
            send_status_ws_message(task_id, 'error')
            return

        try:
            calculated_data = calc(data)
        except Exception as e:
            print('Error (calculate): Failed to calculate results. ' + str(e))
            update_task_status(task_id, 'error')
            send_status_ws_message(task_id, 'error')
            return

        try:
            forecast_period_file = df_to_excel(
                calculated_data['forecast_period'])
            print('successfully calculated forecast_period')
            forecast_forward_file = df_to_excel(
                calculated_data['forecast_forward'])
            print('successfully calculated forecast_forward')
            forecast_common_forward_file = df_to_excel(
                calculated_data['forecast_common_forward'])
            print('successfully calculated forecast_common_forward')
            forecast_common_finance = df_to_excel(
                calculated_data['forecast_common_finance'])
            print('successfully calculated forecast_common_finance')
        except Exception as e:
            print(
                'Error (to excel): Failed to convert calculated data to excel.' + str(e))
            update_task_status(task_id, 'error')
            send_status_ws_message(task_id, 'error')
            return

        ###########
        archive = io.BytesIO()

        with ZipFile(archive, 'w') as zip_archive:
            # Create three files on zip archive

            file1 = ZipInfo('прогноз склада на следующий период.xlsx')
            zip_archive.writestr(file1, forecast_period_file)
            file2 = ZipInfo('прогноз продаж по периодам.xlsx')
            zip_archive.writestr(file2, forecast_forward_file)
            file3 = ZipInfo('накопительный прогноз продаж по периодам.xlsx')
            zip_archive.writestr(file3, forecast_common_forward_file)
            file3 = ZipInfo('накопительный прогноз маржи.xlsx')
            zip_archive.writestr(file3, forecast_common_finance)

        # Flush archive stream to a file on disk
        with open('/tmp/result.zip', 'wb') as f:
            f.write(archive.getbuffer())

        result_object = task_json['zip_url'].replace(
            "request.zip", "result.zip")
        # Upload to Object Storage and generate resigned url
        result_download_url = upload_and_presign(
            '/tmp/result.zip', result_object)
        # Update task status
        complete_task(task_id, result_download_url)
        send_status_ws_message(task_id, 'complete')
    return "OK"
