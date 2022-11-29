import sys
# import cross
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from google.colab import drive # импортирование библиотеки для подключения гугл-диска
from random import sample
import statsmodels.api as sm
import warnings
import scipy
warnings.filterwarnings("ignore")


def calculate(data: dict):
    # drive.mount('/content/gdrive') # связывание гугл-диска с текущей рабочей областью
    # загрузка файла с исходными данными
    df = data['df']
    df.set_index('наименование', inplace=True)
    df = df.transpose().reset_index().transpose()
    #df1 = df.transpose()

    # df1
    # df_input = pd.read_excel('./kipaso_data2_input.xlsx')

    #   df_input['наименование'].to_list()
    #   df_input['Цена закупки'].to_list()
    #   df_input['Маржа'].to_list()

    # In[3]:

    # forc_series=['МВ210-101', 'МУ110-224.16Р М01', 'МУ210-402', 'ПЛК200-01-CS'] # список прогнозируемых позиций
    # analog=['','','',''] # список аналогов (если аналогов нет, то просто список пустых полей)
    # profit=[4496,3726,4509,1000] # список сумм прибыли на единицу соответствующей товарной позиции
    # costs=[12155,10074,12191,10000] # список затрат (хранение+"заморозка денежных средств") на единицу соответствующей товарной позиции
    # список количества месяцев, для которых необходимо провести расчеты по соответствующей товарной позиции
    months = data['months']
    # период начала моделирования с момента последнего периода, на которые есть статистические данные по продажам
    period_start = data['period_start']
    # период для прогнозирования продаж
    forecast_period = data['forecast_period']
    # технический параметр для объема генерации случайных чисел в коде (увеличивает гладкость распределений, но и увеличивает время расчета)
    random_amount = data['random_amount']
    # параметр определения момента окончания моделирования (означает вероятность того, что к моменту окончания моделирования
    failure_chance = data['failure_chance']
    # мы не продадим всю произведенную продукцию)
    # процент стоимости денег для расчета стоимости "затоваривания"
    interest = data['interest']

    # In[4]:

    # список прогнозируемых позиций
    forc_series = data['forc_series']
    # список сумм прибыли на единицу соответствующей товарной позиции
    profit = data['profit']
    # список затрат (хранение+"заморозка денежных средств") на единицу соответствующей товарной позиции
    costs = data['costs']

    # In[5]:

    sales95 = 0  # переменная для определения момента окончания моделирования
    i = 0  # счетчик циклов
    # массив для записи машинно сгенерированных значений продаж
    R_NUM = np.zeros([1, random_amount, len(forc_series)])
    # массив для записи случайных реализаций продаж с учетом продаж по предыдущим периодам
    SUM_SALES = np.zeros([1, random_amount, len(forc_series)])

    for i in range(np.max(months)):  # здесь добавляется количество периодов доп
        # инициализация цикла для прохода по каждой товарной позиции
        for m in range(len(forc_series)):
            # if (analog[m]==''): # в данном условии осуществляем расчет стат базы в зависимости от того, заданы аналоги или нет
            data = df.loc[forc_series[m]].astype('int64').dropna()
            # else:
            # data=df1.loc[analog[m]].astype('int64').dropna()
            if i == 0:
                if len(data) >= 24:  # в данном условии проводится спецификация модели прогноза продаж, если данных больше 31, то добавляется сезонная компонента
                    order = (1, 0, 0)
                    s_order = (1, 0, 0, 12)
                else:
                    order = (1, 0, 0)
                    s_order = (0, 0, 0, 0)
                if m == 0:
                    ORDER = np.array(order).reshape(1, 3)
                    S_ORDER = np.array(s_order).reshape(1, 4)
                else:
                    ORDER = np.concatenate(
                        (ORDER, np.array(order).reshape(1, 3)), 0)
                    S_ORDER = np.concatenate(
                        (S_ORDER, np.array(s_order).reshape(1, 4)), 0)
            model = sm.tsa.statespace.SARIMAX(data, order=tuple(ORDER[m, :]), seasonal_order=tuple(
                S_ORDER[m, :]), trend='c').fit()  # расчет модели по подобранным параметрам
            pred = model.get_forecast(
                steps=i+period_start)  # предсказание продаж
            fd = pred.summary_frame(alpha=0.10)
            if m == 0:
                fd1 = pred.summary_frame(alpha=0.10)
            # запись матожидания будущих продаж
            mu = fd['mean'][len(data)+i+period_start-1]
            # запись СКО будущих продаж
            sigma = fd['mean_se'][len(data)+i+period_start-1]
            # генерация случайных реализация продаж в зависимость от полученных по модели параметров
            r_num = np.round(np.random.normal(mu, sigma, random_amount))
            r_num = np.array(r_num).reshape(1, random_amount)
            # обнуление отрицательных значений полученных продаж
            r_num[(r_num < 0)] = 0
            if i == 0:  # в данном условии происходит запись сгенерированных значений в массив
                R_NUM[0, :, m] = r_num
            else:
                if m == 0:
                    y = np.zeros([1, random_amount, len(forc_series)])
                    R_NUM = np.concatenate([R_NUM, y], axis=0)
                    R_NUM[i, :, m] = r_num
                else:
                    R_NUM[i, :, m] = r_num
            sales = []  # пустой список для записи продаж по периодам
            for k in range(random_amount):
                # случайная генерация накопленных продаж по периодам
                sales.append(sample(list(R_NUM[i, :, m]), 1)[0])
            if i == 0:  # в данном условии происходит запись рассчитанных данных по продажам и прибыли в массивы
                SUM_SALES[0, :, m] = np.array(sales).reshape(1, random_amount)
            else:
                if m == 0:
                    y = np.zeros([1, random_amount, len(forc_series)])
                    SUM_SALES = np.concatenate([SUM_SALES, y], axis=0)
                    SUM_SALES[i, :, m] = np.array(
                        sales).reshape(1, random_amount)
                else:
                    SUM_SALES[i, :, m] = np.array(
                        sales).reshape(1, random_amount)

        i += 1

    q = []
    for m in range(len(forc_series)):
        q.append(np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.01, 1)[-1])
    av = []
    for m in range(len(forc_series)):
        av.append(np.mean(np.cumsum(SUM_SALES[:, :, m], 0), 1)[-1])

    print(f'sum sales1 {np.mean(np.cumsum(SUM_SALES[:,:,m],0),1)}')
    print(i)

    # инициализация цикла (выход из него происходит, если вероятность продажи всего объема произведенных товаров меньше p)
    while i < 12:
        # инициализация цикла для прохода по каждой товарной позиции
        for m in range(len(forc_series)):
            # if (analog[m]==''): # в данном условии осуществляем расчет стат базы в зависимости от того, заданы аналоги или нет
            data = df.loc[forc_series[m]].astype('int64').dropna()
            # else:
            # data=df[analog[m]].dropna()

            model = sm.tsa.statespace.SARIMAX(
                data,
                order=tuple(ORDER[m, :]),
                seasonal_order=tuple(S_ORDER[m, :]),
                trend='c'
            ).fit()  # расчет модели по подобранным параметрам
            pred = model.get_forecast(
                steps=i+period_start)  # предсказание продаж
            fd = pred.summary_frame(alpha=0.10)
            if m == 0:
                fd1 = pred.summary_frame(alpha=0.10)
            # запись матожидания будущих продаж
            mu = fd['mean'][len(data)+i+period_start-1]
            # запись СКО будущих продаж
            sigma = fd['mean_se'][len(data)+i+period_start-1]
            # генерация случайных реализация продаж в зависимость от полученных по модели параметров
            r_num = np.round(np.random.normal(mu, sigma, random_amount))
            r_num = np.array(r_num).reshape(1, random_amount)
            # обнуление отрицательных значений полученных продаж
            r_num[(r_num < 0)] = 0
            if m == 0:
                y = np.zeros([1, random_amount, len(forc_series)])
                R_NUM = np.concatenate([R_NUM, y], axis=0)
                R_NUM[i, :, m] = r_num
            else:
                R_NUM[i, :, m] = r_num
            sales = []  # пустой список для записи продаж по периодам
            for k in range(random_amount):
                # слуайная генерация накопленных продаж по периодам
                sales.append(sample(list(R_NUM[i, :, m]), 1)[0])
            if i == 0:  # в данном условии происходит запись рассчитанных данных по продажам и прибыли в массивы
                SUM_SALES[0, :, m] = np.array(sales).reshape(1, random_amount)
            else:
                if m == 0:
                    y = np.zeros([1, random_amount, len(forc_series)])
                    SUM_SALES = np.concatenate([SUM_SALES, y], axis=0)
                    SUM_SALES[i, :, m] = np.array(
                        sales).reshape(1, random_amount)
                else:
                    SUM_SALES[i, :, m] = np.array(
                        sales).reshape(1, random_amount)
        q = []
        for m in range(len(forc_series)):
            q.append(np.quantile(
                np.cumsum(SUM_SALES[:, :, m], 0), 0.01, 1)[-1])
        i += 1
        # print(i)

    print(f'sum sales2 {np.mean(np.cumsum(SUM_SALES[:,:,m],0),1)}')

    # в данном цикле происходит расчет среднего и медианного прогноза по продажам и прибыли, а также доверительных интервалов
    for m in range(len(forc_series)):
        sales_forecast = np.mean(np.cumsum(SUM_SALES[:, :, m], 0), 1)
        sales_forecast_median = np.median(np.cumsum(SUM_SALES[:, :, m], 0), 1)
        sales_lb = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.025, 1)
        sales_ub = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.975, 1)
        sales_lb1 = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)
        sales_ub1 = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.8, 1)

        # print(sales_forecast)

        r20 = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[months-1]
        r50 = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.5, 1)[months-1]
        r80 = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.8, 1)[months-1]
        r95 = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.95, 1)[months-1]

        t20 = 0
        t50 = 0
        t80 = 0
        t95 = 0

        t_optimum = 0

        if m == 0:  # здесь начинается запись рассчитанных данных в двумерный массив для экспорта данных в эксель
            DATA = pd.DataFrame(sales_forecast, columns=[
                                'Прогноз продаж по позиции "{}"'.format(forc_series[m])])
        else:
            DATA = pd.concat([DATA, pd.DataFrame(sales_forecast, columns=[
                'Прогноз продаж по позиции "{}"'.format(forc_series[m])])], axis=1)
        DATA = pd.concat([DATA, pd.DataFrame(sales_forecast_median, columns=[
            'Медиана прогноза продаж по позиции "{}"'.format(forc_series[m])])], axis=1)
        DATA = pd.concat([DATA, pd.DataFrame(sales_lb, columns=[
            'Нижняя граница прогноза продаж по позиции "{}"'.format(forc_series[m])])], axis=1)
        DATA = pd.concat([DATA, pd.DataFrame(sales_ub, columns=[
            'Верхняя граница прогноза продаж по позиции "{}"'.format(forc_series[m])])], axis=1)
        DATA = pd.concat([DATA, pd.DataFrame(np.mean(SUM_SALES[:, :, m], axis=1), columns=[
            'Прогноз продаж по периодам по позиции "{}"'.format(forc_series[m])])], axis=1)
        DATA = pd.concat([DATA, pd.DataFrame(np.median(SUM_SALES[:, :, m], axis=1), columns=[
            'Медиана прогноза продаж по периодам по позиции "{}"'.format(forc_series[m])])], axis=1)
        DATA = pd.concat([DATA, pd.DataFrame(np.quantile(SUM_SALES[:, :, m], 0.05, axis=1),
                                             columns=['Нижняя граница прогноза продаж по периодам по позиции "{}"'.format(forc_series[m])])], axis=1)
        DATA = pd.concat([DATA, pd.DataFrame(np.quantile(SUM_SALES[:, :, m], 0.95, axis=1),
                                             columns=['Верхняя граница прогноза продаж по периодам по позиции "{}"'.format(forc_series[m])])], axis=1)
        av = np.mean(np.cumsum(SUM_SALES[:, :, m], 0), 1)[months-1]
        # вывод рассчитанных статистик
        print(f'Ожидаемые продажи на {months-1}-й месяц: {np.round(av,2)} шт.')
        # вывод рассчитанных статистик
        print(
            f'Нижняя граница прогноза продаж на {months-1}-й месяц: {np.round(sales_lb[months-1],2)} шт.')
        # вывод рассчитанных статистик
        print(
            f'Верхняя граница прогноза продаж на {months-1}-й месяц: {np.round(sales_ub[months-1],2)} шт.')
        print(f'20% {months-1}-й месяц: {np.round(r20,2)} шт.')
        print(f'50% {months-1}-й месяц: {np.round(r50,2)} шт.')
        print(f'80% {months-1}-й месяц: {np.round(r80,2)} шт.')
        print(f'95% {months-1}-й месяц: {np.round(r95,2)} шт.')

        for k in range(i):
            q = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[k]
            if q > r95:
                q1 = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[k-1]
                t95 = (r95-q1)/(q-q1)+k-1
                break
        # вывод рассчитанных статистик
        print(
            f'Время распродажи с 80% вероятность склада с надежностью 95%: {np.round(t95,2)} месяцев')

        for k in range(i):

            q = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[k]
            if q > r80:
                q1 = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[k-1]
                t80 = (r80-q1)/(q-q1)+k-1
                break
        # вывод рассчитанных статистик
        print(
            f'Время распродажи с 80% вероятность склада с надежностью 80%: {np.round(t80,2)} месяцев')

        for k in range(i):
            q = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[k]
            if q > r50:
                q1 = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[k-1]
                t50 = (r50-q1)/(q-q1)+k-1
                break
        # вывод рассчитанных статистик
        print(
            f'Время распродажи с 80% вероятность склада с надежностью 50%: {np.round(t50,2)} месяцев')

        for k in range(i):
            q = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[k]
            if q > r20:
                q1 = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[k-1]
                t20 = (r20-q1)/(q-q1)+k-1
                break
        # вывод рассчитанных статистик
        print(
            f'Время распродажи с 80% вероятность склада с надежностью 20%: {np.round(t20,2)} месяцев')

        margin_lost20 = (r95-r20)*(0.8 * 0.5)*profit[m]
        margin_lost50 = (r95-r50)*(0.5 * 0.5)*profit[m]
        margin_lost80 = (r95-r80)*(0.2 * 0.5)*profit[m]
        margin_lost95 = 0

        storage_lost95 = (r95-r20)*(t95/12)*interest*costs[m]*0.5
        storage_lost80 = (r80-r20)*(t80/12)*interest*costs[m]*0.5
        storage_lost50 = (r50-r20)*(t50/12)*interest*costs[m]*0.5
        storage_lost20 = 0

        cross_x = [r20, r50, r80, r95]
        cross_xproc = [0.20, 0.50, 0.80, 0.95]
      # cross_margin_lost = [margin_lost20, margin_lost50, margin_lost80, margin_lost95]
      # cross_storage_lost = [storage_lost20, storage_lost50, storage_lost80, storage_lost95]
      # optimum_point = np.round(cross(cross_margin_lost, cross_storage_lost, cross_x),2)
      # print(f'Точка оптимум: {optimum_point}')

        print(f'sum20: {storage_lost20 + margin_lost20}')
        print(f'sum50: {storage_lost50 + margin_lost50}')
        print(f'sum80: {storage_lost80 + margin_lost80}')
        print(f'sum95: {storage_lost95 + margin_lost95}')

        sum_lost20 = storage_lost20 + margin_lost20
        sum_lost50 = storage_lost50 + margin_lost50
        sum_lost80 = storage_lost80 + margin_lost80
        sum_lost95 = storage_lost95 + margin_lost95

        y11 = [sum_lost20, sum_lost50, sum_lost80, sum_lost95]
        x11 = [20, 50, 80, 95]
        f_interp = scipy.interpolate.interp1d(x11, y11, kind='quadratic')
        arr_f = []
        for l in range(20, 96):

            arr_f.append(f_interp(l))

        print(f'суммарные потери: {np.round(min(arr_f),2)}')
        sum_opt = arr_f.index(min(arr_f))+20
        print(f'оптимум покрытия по сумме %: {sum_opt}')

        r_sum_opt = np.quantile(
            np.cumsum(SUM_SALES[:, :, m], 0), (sum_opt/100), 1)[months-1]
        print(f'оптимум покрытия по сумме шт {r_sum_opt}')
        #print(f'оптимум покрытия по пересечению шт {optimum_point}')

        margin95 = r95*profit[m]
        margin80 = r80*profit[m]
        margin50 = r50*profit[m]
        margin20 = r20*profit[m]

        #xproc = np.round(cross(cross_margin_lost, cross_storage_lost, cross_xproc),2)
      # print(f'Точка оптимум в % надежности склада: {xproc}')

        #cost_optimum = optimum_point * costs[m]

        for k in range(i):
            q = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[k]
            if q > r_sum_opt:
                q1 = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[k-1]
                t_sum_opt = (r_sum_opt-q1)/(q-q1)+k-1
                break

        storage_lost_optimum_sum = (r_sum_opt - r20) * \
            (t_sum_opt/12)*interest*costs[m]*0.5

        """  
    for k in range(i):
      q = np.quantile(np.cumsum(SUM_SALES[:,:,m],0),0.2,1)[k]
      if q > optimum_point:
        q1 = np.quantile(np.cumsum(SUM_SALES[:,:,m],0),0.2,1)[k-1]
        t_optimum = (optimum_point-q1)/(q-q1)+k-1
        break      
      
    #storage_lost_optimum = (optimum_point - r20)*(t_optimum/12)*interest*costs[m]*0.5    
    """

        print(
            f'Потери недопродажи при 20% покрытии склада: {np.round(margin_lost20,2)}')
        print(
            f'Потери недопродажи при 50% покрытии склада: {np.round(margin_lost50,2)}')
        print(
            f'Потери недопродажи при 80% покрытии склада: {np.round(margin_lost80,2)}')
        print(
            f'Потери недопродажи при 95% покрытии склада: {np.round(margin_lost95,2)}')

        print(
            f'Потери затовар при 20% покрытии склада: {np.round(storage_lost20,2)}')
        print(
            f'Потери затовар при 50% покрытии склада: {np.round(storage_lost50,2)}')
        print(
            f'Потери затовар при 80% покрытии склада: {np.round(storage_lost80,2)}')
        print(
            f'Потери затовар при 95% покрытии склада: {np.round(storage_lost95,2)}')

        # ======================================================
        # графики
        # plt.figure(figsize=(10, 7))
        # plt.plot(sales_forecast, label='Матожидание')
        # plt.plot(sales_forecast_median, label='Медиана')
        # plt.legend()
        # plt.title(
        #     'Динамика продаж с 95%-ными доверительными интервалами по позиции "{}"'.format(forc_series[m]))
        # plt.xlabel('Время')
        # plt.ylabel('Продажи, шт.')
        # plt.fill_between(np.arange(0, i, 1), sales_lb,
        #                  sales_ub, color='k', alpha=.15)
        # plt.fill_between(np.arange(0, i, 1), sales_lb1,
        #                  sales_ub1, color='k', alpha=.2)
        # plt.show()
        # print(model.summary())
        # plt.figure(figsize=(10, 7))
        # plt.title(
        #     'Динамика продаж по периодам с 95%-ными доверительными интервалами по позиции "{}"'.format(forc_series[m]))
        # plt.plot(np.mean(SUM_SALES[:, :, m], axis=1))
        # plt.plot(np.median(SUM_SALES[:, :, m], axis=1))
        # # plt.plot(np.quantile(SUM_SALES[:,:,m],xproc,axis=1))
        # plt.fill_between(np.arange(0, i, 1), np.quantile(SUM_SALES[:, :, m], 0.05, axis=1), np.quantile(
        #     SUM_SALES[:, :, m], 0.95, axis=1), color='k', alpha=.15)
        # plt.fill_between(np.arange(0, i, 1), np.quantile(SUM_SALES[:, :, m], 0.2, axis=1), np.quantile(
        #     SUM_SALES[:, :, m], 0.8, axis=1), color='k', alpha=.2)
        # plt.xlabel('Время')
        # plt.ylabel('Продажи, шт.')
        # plt.show()
        # ======================================================

        # запись совокупных характеристик в массив для экспорта
        print(f'запись совокупных характеристик в массив для экспорта')
        DATA = pd.concat([DATA, pd.DataFrame(sales_forecast, columns=[
            'Прогноз продаж по всем позициям'])], axis=1)
        print(f'Прогноз продаж по всем позициям')
        DATA = pd.concat([DATA, pd.DataFrame(sales_forecast_median, columns=[
            'Медиана прогноза продаж по всем позициям'])], axis=1)
        print(f'Медиана прогноза продаж по всем позициям')
        DATA = pd.concat([DATA, pd.DataFrame(sales_lb, columns=[
            'Нижняя граница прогноза продаж по всем позициям'])], axis=1)
        print(f'Нижняя граница прогноза продаж по всем позициям')
        DATA = pd.concat([DATA, pd.DataFrame(sales_ub, columns=[
            'Верхняя граница прогноза продаж по всем позициям'])], axis=1)
        print(f'Верхняя граница прогноза продаж по всем позициям')
        DATA = pd.concat([DATA, pd.DataFrame(np.mean(SUM_SALES[:, :, m], axis=1), columns=[
            'Прогноз продаж по периодам по всем позициям'])], axis=1)
        print(f'Прогноз продаж по периодам по всем позициям')
        DATA = pd.concat([DATA, pd.DataFrame(np.median(SUM_SALES[:, :, m], axis=1), columns=[
            'Медиана прогноза продаж по периодам по всем позициям'])], axis=1)
        print(f'Медиана прогноза продаж по периодам по всем позициям')
        DATA = pd.concat([DATA, pd.DataFrame(np.quantile(SUM_SALES[:, :, m], 0.025, axis=1),
                                             columns=['Нижняя граница прогноза продаж по периодам по всем позициям'])], axis=1)
        print(f'Нижняя граница прогноза продаж по периодам по всем позициям')
        DATA = pd.concat([DATA, pd.DataFrame(np.quantile(SUM_SALES[:, :, m], 0.975, axis=1),
                                             columns=['Верхняя граница прогноза продаж по периодам по всем позициям'])], axis=1)
        print(f'Верхняя граница прогноза продаж по периодам по всем позициям')

        print(f'Данные подсчитаны')

        vivod_na_period = ['80% вер. продаж шт',
                           '50% вер. продаж шт',
                           '20% вер. продаж шт',
                           '5% вер. продаж шт',
                           '80% вер. маржа',
                           '50% вер. маржа',
                           '20% вер. маржа',
                           '5% верю. маржа',
                           'Оптимальный склад шт',
                           'Надежность оптимального склада %',
                           'Стоимость закупки оптим. склада',
                           'Макс. время распродажи оптим. склада',
                           'Потери на затовар. оптим. склада',
                           'оптимум покрытия по сумме %',
                           'оптимум покрытия по сумме шт',
                           'Макс. время распродажи оптим. склада по сум',
                           'Потери на затовар. оптим. склада по сум'
                           ]

        data_na_period = np.round([r20, r50, r80, r95, margin20, margin50, margin80, margin95,
                                  0, 0, 0, 0, 0,
                                  sum_opt, r_sum_opt, t_sum_opt, storage_lost_optimum_sum], 2)
        print(f'data_na_period подсчитаны')
        if m == 0:  # здесь начинается запись рассчитанных данных в двумерный массив для экспорта данных в эксель
            DATA1 = pd.DataFrame(data_na_period, columns=[
                forc_series[m]], index=vivod_na_period)
        else:
            DATA1 = pd.concat([DATA1, pd.DataFrame(data_na_period, columns=[
                              forc_series[m]], index=vivod_na_period)], axis=1)

        #DATA1=pd.concat([DATA1,pd.DataFrame([np.round(r50,2)] ,columns=[forc_series[m]], index = ['вероятность продать 50%, не менее шт, (надежность склада 50%)'])], axis=0)
        #DATA1=pd.concat([DATA1,pd.DataFrame([np.round(r80,2)] ,columns=[forc_series[m]], index = ['вероятность продать 20%, не менее шт, (надежность склада 80%)'])], axis=0)
        #DATA1=pd.concat([DATA1,pd.DataFrame([np.round(r95,2)] ,columns=[forc_series[m]], index = ['вероятность продать 5%, не менее шт, (надежность склада 95%)'])], axis=0)

        if m == 0:  # здесь начинается запись рассчитанных данных в двумерный массив для экспорта данных в эксель
            DATA2 = pd.DataFrame(np.round(np.mean(SUM_SALES[:, :, m], axis=1), 2), columns=[
                'Прогноз продаж по периодам по позиции "{}"'.format(forc_series[m])])
        else:
            DATA2 = pd.concat([DATA2, pd.DataFrame(np.round(np.mean(SUM_SALES[:, :, m], axis=1), 2), columns=[
                              'Прогноз продаж по периодам по позиции "{}"'.format(forc_series[m])])], axis=1)

        DATA2 = pd.concat([DATA2, pd.DataFrame(np.quantile(SUM_SALES[:, :, m], 0.2, axis=1),
                                               columns=['80% вер. Нижняя граница прогноза продаж по периодам по позиции "{}"'.format(forc_series[m])])], axis=1)
        DATA2 = pd.concat([DATA2, pd.DataFrame(np.quantile(SUM_SALES[:, :, m], 0.8, axis=1),
                                               columns=['20% вер. Верхняя граница прогноза продаж по периодам по позиции "{}"'.format(forc_series[m])])], axis=1)
        print(f'DATA подсчитаны')
        # DATA2=pd.concat([DATA2,pd.DataFrame(np.quantile(SUM_SALES[:,:,m],xproc,axis=1),
        # columns=['Оптимальный склад по периодам по позиции "{}"'.format(forc_series[m])])],axis=1)


    result = dict()

    result['forecast'] = DATA
    result['testnewforecastperiod'] = DATA1.transpose()
    result['testnewforecastforward'] = DATA2.transpose()

    # экспорт сформированной базы в эксель
    # DATA.to_excel('./forecast_results.xlsx')
    # DATA1 = DATA1.transpose()
    # экспорт сформированной базы в эксель на период
    # DATA1.to_excel('./testnewforecastperiod_results.xlsx')
    # DATA2 = DATA2.transpose()
    # экспорт сформированной базы в эксель на период
    # DATA2.to_excel('./testnewforecastforward_results.xlsx')

    # как выбрать N, что такое альфа, как работает расчет времени распродажи, как задать точное колво периодов прогноза

    return result


sys.modules[__name__] = calculate
