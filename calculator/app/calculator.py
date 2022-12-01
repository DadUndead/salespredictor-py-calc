import json
import os
from zipfile import ZipInfo, ZipFile

import ydb
import boto3

import time
import data_parse
import calculate as calc
import io
import pandas as pd


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
    df.to_excel(excel_writer=writer, index=False, sheet_name='Sheet1')
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


def handle_process_event(event, context):
    

    for message in event['messages']:
        task_json = json.loads(message['details']['message']['body'])
        task_id = task_json['task_id']
        zip_url = task_json['zip_url']

        print(f'Processing task. task_id:{task_id} zip_url:{zip_url}')

        # Update task status
        start_processing_task(task_id)

        print('Unzipping request archive...')
        unzip_archive(zip_url, '/tmp')

        # Calculate result

        try:
            data = data_parse('/tmp/data.xlsx', '/tmp/data_input.xlsx')
        except Exception as e:
            print(f'Error (data_parse): Failed to parse data files. ' + str(e))
            return

        try:
            calculated_data = calc(data)
        except Exception as e:
            print('Error (calculate): Failed to calculate results. ' + str(e))
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
        except Exception as e:
            print(
                'Error (to excel): Failed to convert calculated data to excel.' + str(e))
            return

        ###########
        archive = io.BytesIO()

        with ZipFile(archive, 'w') as zip_archive:
            # Create three files on zip archive

            file1 = ZipInfo('forecast_period_results.xlsx')
            zip_archive.writestr(file1, forecast_period_file)
            file2 = ZipInfo('forecast_forward_results.xlsx')
            zip_archive.writestr(file2, forecast_forward_file)
            file3 = ZipInfo('forecast_common_forward_results.xlsx')
            zip_archive.writestr(file3, forecast_common_forward_file)

        # Flush archive stream to a file on disk
        with open('/tmp/result.zip', 'wb') as f:
            f.write(archive.getbuffer())

        result_object = task_json['zip_url'].replace(
            "request.zip", "result.zip")
        # Upload to Object Storage and generate presigned url
        result_download_url = upload_and_presign(
            '/tmp/result.zip', result_object)
        # Update task status
        complete_task(task_id, result_download_url)
    return "OK"
