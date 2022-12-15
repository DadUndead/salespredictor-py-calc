import sys
import scipy
import numpy as np
import pandas as pd
from random import sample
import statsmodels.api as sm
from scipy.special import logsumexp
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def calculate(data: dict):

    df = data['df']  # загрузка исходных данных продаж
    df.set_index('наименование', inplace=True)
    df = df.transpose().reset_index().transpose()

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

    # список прогнозируемых позиций
    forc_series = data['forc_series']
    # список сумм прибыли на единицу соответствующей товарной позиции
    profit = data['profit']
    # список затрат (хранение+"заморозка денежных средств") на единицу соответствующей товарной позиции
    costs = data['costs']
    # маржа доля и куммулятивная сумма
    margin_part = data['margin_part']
    margin_cum = data['margin_cum']

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
                    s_order = (1, 0, 0, forecast_period)
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

    # инициализация цикла (выход из него происходит по периоду прогноза)
    while i < forecast_period:
        # инициализация цикла для прохода по каждой товарной позиции
        for m in range(len(forc_series)):
            # if (analog[m]==''): # в данном условии осуществляем расчет стат базы в зависимости от того, заданы аналоги или нет
            data = df.loc[forc_series[m]].astype('int64').dropna()
            # else:
            # data=df[analog[m]].dropna()

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

    # в данном цикле происходит расчет среднего и медианного прогноза по продажам и прибыли, а также доверительных интервалов
    for m in range(len(forc_series)):
        sales_forecast = np.mean(np.cumsum(SUM_SALES[:, :, m], 0), 1)
        sales_forecast_median = np.median(np.cumsum(SUM_SALES[:, :, m], 0), 1)
        sales_lb = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.025, 1)
        sales_ub = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.975, 1)
        sales_lb1 = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)
        sales_ub1 = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.8, 1)

        r20 = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[months-1]
        r50 = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.5, 1)[months-1]
        r80 = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.8, 1)[months-1]
        r95 = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.95, 1)[months-1]

        # блок сигмы тренд
        data = df.loc[forc_series[m]].astype('int64').dropna()
        regressor = LinearRegression()
        regressor.fit(np.array(data.index.to_list()).reshape((-1, 1)), data)

        if data.mean() > 0:
            trend = regressor.coef_[0] * forecast_period / data.mean() * 100  # тренд % рост продаж за цикл
        else:
            trend = 0

        y_pred = regressor.predict(np.array(data.index.to_list()).reshape((-1, 1)))
        y_test = data

        sigma_trend = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

        if sigma_trend > 0:
            vari = sigma_trend / data.mean()
        else:
            vari = 99999

        XYZ = ''

        if vari < 0.3:
            XYZ = 'X'
        elif vari < 0.7:
            XYZ = 'Y'
        else:
            XYZ = 'Z'

        ABC = ''

        if margin_cum[m] < 80:
            ABC = 'A'
        elif margin_cum[m] < 95:
            ABC = 'B'
        else:
            ABC = 'C'



        t20 = 0
        t50 = 0
        t80 = 0
        t95 = 0

        t_optimum = 0
        t_sum_opt = 0

        # расчет экстраполированных продаж для расчета долгоокупаемых складов

        if (np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[i-1]) > 0:
            r_month = (np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[i-1]) / forecast_period
        else:
            r_month = 99999



        if np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[i-1] > r95:
            for k in range(i):
                q = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[k]
                if q > r95:
                    q1 = np.quantile(
                        np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[k-1]
                    t95 = (r95-q1)/(q-q1)+k-1
                    break
        else:
            t95 = r95 / r_month

        if np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[i-1] > r80:

            for k in range(i):

                q = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[k]
                if q > r80:
                    q1 = np.quantile(
                        np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[k-1]
                    t80 = (r80-q1)/(q-q1)+k-1
                    break
        else:
            t80 = r80 / r_month

        if np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[i-1] > r50:

            for k in range(i):
                q = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[k]
                if q > r50:
                    q1 = np.quantile(
                        np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[k-1]
                    t50 = (r50-q1)/(q-q1)+k-1
                    break
        else:
            t50 = r50 / r_month

        if np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[i-1] > r20:
            for k in range(i):
                q = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[k]
                if q > r20:
                    q1 = np.quantile(
                        np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[k-1]
                    t20 = (r20-q1)/(q-q1)+k-1
                    break

        else:
            t20 = r20 / r_month

        margin_lost20 = (r95-r20)*(0.8 * 0.5)*profit[m]
        margin_lost50 = (r95-r50)*(0.5 * 0.5)*profit[m]
        margin_lost80 = (r95-r80)*(0.2 * 0.5)*profit[m]
        margin_lost95 = 0

        storage_lost95 = (r95-r20)*(t95/12)*interest*costs[m]*0.5
        storage_lost80 = (r80-r20)*(t80/12)*interest*costs[m]*0.5
        storage_lost50 = (r50-r20)*(t50/12)*interest*costs[m]*0.5
        storage_lost20 = 0

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

        sum_opt = arr_f.index(min(arr_f))+20

        r_sum_opt = np.quantile(
            np.cumsum(SUM_SALES[:, :, m], 0), (sum_opt/100), 1)[months-1]

        margin95 = np.round(r95,0)*profit[m]
        margin80 = np.round(r80,0)*profit[m]
        margin50 = np.round(r50,0)*profit[m]
        margin20 = np.round(r20,0)*profit[m]

        costs95 = np.round(r95,0) * costs[m]
        costs80 = np.round(r80,0) * costs[m]
        costs50 = np.round(r50,0) * costs[m]
        costs20 = np.round(r20,0) * costs[m]
        costs_opt = np.round(r_sum_opt,0) * costs[m]


        if np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[i-1] > r_sum_opt:
            for k in range(i):
                q = np.quantile(np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[k]
                if q > r_sum_opt:
                    q1 = np.quantile(
                        np.cumsum(SUM_SALES[:, :, m], 0), 0.2, 1)[k-1]
                    t_sum_opt = (r_sum_opt-q1)/(q-q1)+k-1
                    break
        else:

            t_sum_opt = r_sum_opt / r_month

        storage_lost_optimum_sum = (
            r_sum_opt - r20)*(t_sum_opt/12)*interest*costs[m]*0.5

        vivod_na_period = ['ABC',
                           'XYZ',
                           'Доля маржи %',
                            'Коэфф. вариации',
                           'тренд %',
                           'Оптимальный склад, шт',
                           'Макс. время распродажи оптим. склада',
                           '80% вер. продаж шт',
                           '50% вер. продаж шт',
                           '20% вер. продаж шт',
                           '5% вер. продаж шт',
                           '80% вер. маржа',
                           '50% вер. маржа',
                           '20% вер. маржа',
                           '5% вер. маржа',
                           'Затраты на закупку оптим. склада',
                           'Затраты на закупку по 80% вер',
                           'Затраты на закупку по 50% вер',
                           'Затраты на закупку по 20% вер',
                           'Затраты на закупку по 5% вер'
                           ]

        data_na_period = [ABC,  XYZ, np.round(margin_part[m], 1), np.round(vari, 2), np.round(trend, 1), np.round(r_sum_opt, 0), np.round(t_sum_opt, 1), np.round(r20, 0), np.round(r50, 0),
                          np.round(r80, 0), np.round(r95, 0),
                          np.round(margin20, 0), np.round(margin50, 0), np.round(margin80, 0), np.round(margin95, 0),
                          np.round(costs_opt, 0), np.round(costs20, 0),
                          np.round(costs50, 0), np.round(costs80, 0), np.round(costs95, 0)]

        if m == 0:  # здесь начинается запись рассчитанных данных в двумерный массив для экспорта данных в эксель
            DATA1 = pd.DataFrame(data_na_period, columns=[
                                 forc_series[m]], index=vivod_na_period)
        else:
            DATA1 = pd.concat([DATA1, pd.DataFrame(data_na_period, columns=[
                              forc_series[m]], index=vivod_na_period)], axis=1)

        if m == 0:  # здесь начинается запись рассчитанных данных в двумерный массив для экспорта данных в эксель
            DATA2 = pd.DataFrame(np.round(np.quantile(SUM_SALES[:, :, m], 0.8, axis=1), 0), columns=[
                '20% вероят. Верхняя граница прогноза продаж по периодам по позиции "{}"'.format(forc_series[m])])
        else:
            DATA2 = pd.concat([DATA2, pd.DataFrame(np.round(np.quantile(SUM_SALES[:, :, m], 0.8, axis=1), 0),
                                                   columns=['20% вероят. Верхняя граница прогноза продаж по периодам по позиции "{}"'.format(
                                                           forc_series[m])])], axis=1)

        DATA2 = pd.concat([DATA2, pd.DataFrame(np.round(np.quantile(SUM_SALES[:, :, m], 0.5, axis=1), 0),
                                               columns=['Средний прогноз продаж по периодам по позиции "{}"'.format(
                                                   forc_series[m])])], axis=1)
        DATA2 = pd.concat([DATA2, pd.DataFrame(np.round(np.quantile(SUM_SALES[:, :, m], 0.2, axis=1), 0),
                                               columns=['80% вероят. Нижняя граница прогноза продаж по периодам по позиции "{}"'.format(
                                                       forc_series[m])])], axis=1)

        # np.round(np.mean(SUM_SALES[:,:,m],axis=1),2)

    # np.quantile(SUM_SALES[:,:,m],0.2,axis=1)

    # np.quantile(SUM_SALES[:,:,m],0.8,axis=1)

        if m == 0:  # здесь начинается запись рассчитанных данных в двумерный массив для экспорта данных в эксель
            DATA3 = pd.DataFrame(np.round(sales_ub1,0),
                                 columns=['"{}" 20% вероят. Верхняя граница прогноза накопительных продаж по позиции'.format(forc_series[m])])
        else:
            DATA3 = pd.concat([DATA3, pd.DataFrame(np.round(sales_ub1,0),
                                                   columns=['"{}" 20% вероят. Верхняя граница прогноза накопительных продаж по позиции'.format(forc_series[m])])], axis=1)

        DATA3 = pd.concat([DATA3, pd.DataFrame(np.round(sales_forecast_median,0),
                                               columns=['"{}" Средний прогноз накопительных продаж по позиции'.format(forc_series[m])])], axis=1)
        DATA3 = pd.concat([DATA3, pd.DataFrame(np.round(sales_lb1,0),
                                               columns=['"{}" 80% вероят. Нижняя граница прогноза накопительных продаж по позиции'.format(forc_series[m])])], axis=1)

    result = dict()

    # экспорт сформированной базы на 1 период
    result['forecast_period'] = DATA1.transpose()
    # экспорт сформированной базы в прогноз продаж по периодам вперед
    result['forecast_forward'] = DATA2.transpose()
    # экспорт сформированной базы в прогноз накопительных продаж вперед
    result['forecast_common_forward'] = DATA3.transpose()

    return result

sys.modules[__name__] = calculate

# как выбрать N, что такое альфа, как работает расчет времени распродажи, как задать точное колво периодов прогноза
