import json
import os
import uuid
from urllib.parse import urlencode

import zipfile

import boto3
import requests

import data_parse
import calculate as calc

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

# Converter handler


def unzip_archive(zip_url, dest):
    client = get_storage_client()
    bucket = os.environ['BUCKET_NAME']
    path_to_zip_file = './tmp/request.zip'
    client.download_file(zip_url, bucket, path_to_zip_file)
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(dest)


def upload_and_presign(file_path, object_name):
    client = get_storage_client()
    client.upload_file(file_path, os.environ['BUCKET_NAME'], object_name)
    return client.generate_presigned_url('get_object', Params={'Bucket': bucket, 'Key': object_name}, ExpiresIn=3600)


def df_to_excel(df):
    # Creating output and writer (pandas excel writer)
    out = io.BytesIO()
    writer = pd.ExcelWriter(out, engine='xlsxwriter')

    # Export data frame to excel
    df.to_excel(excel_writer=writer, index=False, sheet_name='Sheet1')
    writer.save()
    writer.close()

    return out.getvalue()


def handle_process_event(event, context):
    for message in event['messages']:
        task_json = json.loads(message['details']['message']['body'])
        task_id = task_json['task_id']

        print('Unzipping request archive...')
        unzip_archive(task_json['zip_url'], './tmp/unzipped')

        # Calculate result

        try:
            data = data_parse('./tmp/data.xlsx', './tmp/data_input,xlsx')
        except:
            print('Error (data_parse): Failed to parse data files.')
            return

        try:
            calculated_data = calc(data)
        except:
            print('Error (calculate): Failed to calculate results.')
            return

        try:
            forecast_period_file = df_to_excel(calculated_data['forecast_period'])
            print('successfully calculated forecast_period')
            forecast_forward_file = df_to_excel(calculated_data['forecast_forward'])
            print('successfully calculated forecast_forward')
            forecast_common_forward_file = df_to_excel(calculated_data['forecast_common_forward'])
            print('successfully calculated forecast_common_forward')
        except:
            print('Error (to excel): Failed to convert calculated data to excel.')
            return

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "a",
                             zipfile.ZIP_DEFLATED, False) as zip_file:
            for file_name, data in [('forecast_period_results.xlsx', forecast_period_file),
                                    ('forecast_forward_results.xlsx',
                                     forecast_forward_file),
                                    ('forecast_common_forward_results.xlsx', forecast_common_forward_file)]:
                zip_file.writestr(file_name, data.getvalue())

        zip_file.write('./tmp/result.zip')
        zip_file.close()

        result_object = f"result-{task_id}.zip"
        # Upload to Object Storage and generate presigned url
        result_download_url = upload_and_presign(
            '/tmp/result.zip', result_object)
        # Update task status in DocAPI
        get_docapi_table().update_item(
            Key={'task_id': task_id},
            AttributeUpdates={
                'status': {'Value': 'complete', 'Action': 'PUT'},
                'result_url': {'Value': result_download_url, 'Action': 'PUT'},
            }
        )
    return "OK"
