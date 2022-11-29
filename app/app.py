from os import environ
from flask import Flask, request, abort, jsonify
import jwt
import io
import asyncio

import pandas as pd
import data_parse
import calculate as calc
import get_token
from s3 import S3Client
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)


@app.route('/api/hello-python', methods=['GET'])
def hello_world():
    return 'Hello user!'

async def df_to_excel(df):
    # Creating output and writer (pandas excel writer)
    out = io.BytesIO()
    writer = pd.ExcelWriter(out, engine='xlsxwriter')

    # Export data frame to excel
    df.to_excel(excel_writer=writer, index=False, sheet_name='Sheet1')
    writer.save()
    writer.close()

    return out.getvalue()


async def process_files_async():

    token = get_token(request.headers['Authorization'])
    decoded_token = jwt.decode(token, environ['TOKEN_KEY'],  algorithms=["HS256"])
    user_id = decoded_token['user_id']

    request_id = request.json['requestId']
    data_file = request.json['dataFile']
    data_file_input = request.json['dataFileInput']

    print(f'user_id: {user_id}')
    print(f'request_id: {request_id}')
    print(f'data_file: {data_file}')
    print(f'data_file_input: {data_file_input}')

    s3 = S3Client(
        access_key=environ['SA_ACCESS_KEY_ID'],
        secret_key=environ['SA_ACCESS_SECRET_KEY'],
        region=environ['BUCKET_REGION'],
        s3_bucket=environ['BUCKET_NAME']
    )

    print(f'starting process!')

    data_url = s3.signed_download_url(
        f'storage/{user_id}/requests/{request_id}/{data_file}', max_age=300)
    data_input_url = s3.signed_download_url(
        f'storage/{user_id}/requests/{request_id}/{data_file_input}', max_age=300)

    print(f'data=>{data_url}')
    print(f'data_input=>{data_input_url}')

    try:
        data = data_parse(data_url, data_input_url)
    except:
        print('Error (data_parse): Failed to get or parse files from s3.')
        return abort(400)
    
    print(f'parsed data->{data}')

    try:
        calculated_data = calc(data)
    except:
        print('Error (calculate): Failed to calculate results.')

    forecast_file = await df_to_excel(calculated_data['forecast'], )
    testnewforecastperiod_file = await df_to_excel(calculated_data['testnewforecastperiod'], )
    testnewforecastforward_file = await df_to_excel(calculated_data['testnewforecastforward'], )

    try:
        await s3.upload(f'storage/{user_id}/requests/{request_id}/forecast_results.xlsx', forecast_file)
        await s3.upload(f'storage/{user_id}/requests/{request_id}/testnewforecastperiod_results.xlsx', testnewforecastperiod_file)
        await s3.upload(f'storage/{user_id}/requests/{request_id}/testnewforecastforward_results.xlsx', testnewforecastforward_file)
    except:
        print('Error (upload results): Failed to save result files on s3.')
        return abort(400)

    return jsonify(success=True)


@app.route('/api/calculate', methods=['POST'])
def process_files():
    return asyncio.run(process_files_async())


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=environ['PORT'])
