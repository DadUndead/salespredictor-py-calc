# Облачные функции для расчета

# calculate-api
    Принимает json
    {
        "action" : "convert" | "get_task_status"
        "task_id" : string, // номер задачи
        "zip_url" : string, // адрес архива с исходными файлами на s3
    }
    
    action: "get_task_status" - возвращает идентификатор задачи
    action: "convert" - добавляет задачу в очередь sqs на исполнение

# calculator
    Срабатывает по триггеру из очереди sqs
    Принимает данные в формате
    {
        "task_id" : string, // номер задачи
        "zip_url" : string, // адрес архива с исходными файлами на s3
    }
    Обрабатывает файлы и добавляет в таблицу `requests` ссылку (s3) на архив в формате result.zip