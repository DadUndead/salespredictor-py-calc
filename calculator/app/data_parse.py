import sys
import pandas as pd

def data_parse(data_file_url, data_file_input_url):

    # загрузка файла с исходными данными
    # df = pd.read_excel('./kipaso_data2.xlsx')
    # df_input = pd.read_excel('./kipaso_data2_input.xlsx')
    
    df = pd.read_excel(data_file_url)
    df_input = pd.read_excel(data_file_input_url)
    df_input = df_input[df_input.columns[:3]]

    forecast_period = 12 #цикл периодичности

    #нан убираем и меняем на 0
    df = df.dropna(axis =0, how='all')
    df = df.dropna(axis=1, how='all')
    df_input = df_input.dropna(axis=0, how='all')
    df_input = df_input.dropna(axis=1, how='all')
    df = df.fillna(0)
    df_input = df_input.fillna(0)

    #делаем мердж двух таблиц по inner и дропаем дубликаты. Далее обращения идут по номеру столбца а не названию
    df_mid = df_input.merge(df, left_on=df_input.columns[0], right_on=df.columns[0], how='inner', copy='False')
    df_data = df_mid.drop_duplicates(df_mid.columns[0])



    df_data['маржа_сум'] = df_data[df_data.columns[3:]][df_data[df_data.columns[3:]].columns[-1*forecast_period:]].sum(axis=1) * df_data['Маржа'] #сумм маржа за посл 12 периодов


    df_data['Доля маржи'] = df_data['маржа_сум'] * 100 / df_data['маржа_сум'].sum() #доля маржи %
    df_data = df_data.sort_values(by='Доля маржи', ascending=False)
    df_data['cumsum'] = df_data['Доля маржи'].cumsum() # накомпленная доля для АБС
    df_data = df_data.sort_index() #возвращаем порядок



    data = dict()

    data['df'] = df_data.drop(['маржа_сум', 'Доля маржи', 'cumsum', df_data.columns[[1,2]][0], df_data.columns[[1,2]][1]], axis = 1)

    # forc_series=['МВ210-101', 'МУ110-224.16Р М01', 'МУ210-402', 'ПЛК200-01-CS'] # список прогнозируемых позиций
    # analog=['','','',''] # список аналогов (если аналогов нет, то просто список пустых полей)
    # profit=[4496,3726,4509,1000] # список сумм прибыли на единицу соответствующей товарной позиции
    # costs=[12155,10074,12191,10000] # список затрат (хранение+"заморозка денежных средств") на единицу соответствующей товарной позиции
    
    data['months'] = 1  # список количества месяцев, для которых необходимо провести расчеты по соответствующей товарной позиции
    data['period_start'] = 1  # период начала моделирования с момента последнего периода, на которые есть статистические данные по продажам
    data['forecast_period'] = forecast_period  # период для прогнозирования продаж

    # технический параметр для объема генерации случайных чисел в коде (увеличивает гладкость распределений, но и увеличивает время расчета)
    data['random_amount'] = 200
    # параметр определения момента окончания моделирования (означает вероятность того, что к моменту окончания моделирования
    data['failure_chance'] = 0.025

    # мы не продадим всю произведенную продукцию)
    data['interest'] = 0.3  # процент стоимости денег для расчета стоимости "затоваривания"

    # список прогнозируемых позиций
    data['forc_series'] = df_data[df_data.columns[0]].to_list() # наименование
    # список сумм прибыли на единицу соответствующей товарной позиции
    data['profit'] = df_data[df_data.columns[2]].to_list() # маржа
    # список затрат (хранение+"заморозка денежных средств") на единицу соответствующей товарной позиции
    data['costs'] = df_data[df_data.columns[1]].to_list() # цена закупки

    data['margin_part'] = df_data['Доля маржи'].to_list()
    data['margin_cum'] = df_data['cumsum'].to_list()


    return data

sys.modules[__name__] = data_parse