import os
from flask import Flask, request, jsonify, send_file, make_response
import io

import pandas as pd
import data_parse
import calculate as calc

app = Flask(__name__)


@app.route('/api/hello-python')
def hello_world():
    return 'Hello user!'


@app.route('/api/calculate', methods=['POST'])
def calculate():
    if request.method == 'POST':
        data_url = request.json['data']
        data_input_url = request.json['data_input']

        data = data_parse(data_url, data_input_url)
        calculated_data = calc(data)
        
        # Creating output and writer (pandas excel writer)
        out = io.BytesIO()
        writer = pd.ExcelWriter(out, engine='xlsxwriter')

        # Export data frame to excel
        calculated_data.to_excel(excel_writer=writer, index=False, sheet_name='Sheet1')
        writer.save()
        writer.close()

        # Flask create response 
        resp = make_response(out.getvalue())
        resp.headers["Content-Disposition"] = "attachment; filename=export.xlsx"
        resp.headers["Content-Type"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        return resp
    return 'Failed'


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
