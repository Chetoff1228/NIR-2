import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from tqdm import tqdm
from multiprocessing import Pool
from sklearn import preprocessing
from collections import defaultdict
import os
from datetime import datetime
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def get_index_data_len(n, df):
    return sum(1 for i in df.index if i[1] == n)


def get_index_dict(index_list, l):
    index_dict = defaultdict(list)
    for i in index_list:
        index_dict[i[l]].append(i[1])
    return index_dict

def diff_share(dict_n0, dict_n1):
    sum_0 = sum(dict_n0.values())
    sum_1 = sum(dict_n1.values())
    k_sum = min(sum_0, sum_1) / max(sum_0, sum_1)
    dict_diff = {}

    for i in dict_n1:
        if i in dict_n0:
            dict_diff[i] = k_sum * (dict_n1[i] - dict_n0[i])
        else:
            dict_diff[i] = k_sum * dict_n1[i]

    for i in dict_n0:
        if i not in dict_diff:
            dict_diff[i] = -k_sum * dict_n0[i]

    return dict_diff


def get_corr_all(index_fund_list):
    id_ex = set(cbonds_df.id)
    shape_price_diff_list = []
    diff_shape_mean_list = []
    
    for index_fund in index_fund_list:
        if index_fund not in id_ex:
            return 'stop'
        
        name_fund = cbonds_df.loc[cbonds_df.id == index_fund, 'name'].iloc[0]
        data_fund = set(pai_df.loc[pai_df['Фонд'] == name_fund, 'Дата']) & set(result[index_fund == id_ex][0])
        data_fund = sorted(data_fund)
        
        shape_price = pai_df.loc[(pai_df['Фонд'] == name_fund) & pai_df['Дата'].isin(data_fund), 'Пай'].tolist()
        shape_price_diff = (shape_price[-1] - shape_price[0]) / shape_price[0]
        
        diff_shape = result[(index_fund == id_ex)][0]
        diff_shape = diff_shape[np.isin(diff_shape, data_fund)]
        diff_shape_mean = diff_shape.mean()
        
        shape_price_diff_list.append(shape_price_diff)
        diff_shape_mean_list.append(diff_shape_mean)
    
    diff_shape_norm = preprocessing.normalize([diff_shape_mean_list])
    shape_price_norm = preprocessing.normalize([shape_price_diff_list])
    r = np.corrcoef(diff_shape_norm, shape_price_norm)
    
    return r[0][1], np.array(diff_shape_mean_list), np.array(shape_price_diff_list)


def get_corr(index_fund):
    id_ex = set(cbonds_df.id)
    if index_fund not in id_ex:
        return 'stop'
    
    name_fund = cbonds_df.loc[cbonds_df.id == index_fund, 'name'].iloc[0]
    data_fund = np.array(result[id_ex == index_fund])
    
    data_fund = list(set(data_fund))
    data_fund = sorted(set(pai_df.loc[(pai_df.Фонд == name_fund) & pai_df.Дата.isin(data_fund), 'Дата']))
    
    shape_price = np.array(pai_df.loc[(pai_df.Фонд == name_fund) & pai_df.Дата.isin(data_fund), 'Пай'])
    diff_shape = np.array(result[(id_ex == index_fund) & np.isin(result, data_fund)])
    
    diff_shape_norm = preprocessing.normalize([diff_shape])
    shape_price_norm = preprocessing.normalize([shape_price])
    
    r = np.corrcoef(diff_shape_norm, shape_price_norm)
    return r[0][1]


def calculate_structure_difference(dict1, dict2):
    total1 = sum(dict1.values())
    total2 = sum(dict2.values())
    total_diff = total2 - total1

    difference = {}
    for key, value1 in dict1.items():
        value2 = dict2.get(key, 0)
        diff = value2 - value1 + (value1 / total1) * total_diff
        difference[key] = abs(diff)

    for key, value2 in dict2.items():
        if key not in dict1:
            difference[key] = value2

    return difference

def abs_sum(d):
    return sum(abs(v) for v in d.values())


def change_share(id):
    data_dict = {}
    data_list = sorted(set(cbonds_df[cbonds_df.id == id].date))

    for t0, t1 in zip(data_list, data_list[1:]):
        data_t0 = dict(zip(cbonds_df[(cbonds_df.id == id) & (cbonds_df.date == t0)].sec_isin, cbonds_df[(cbonds_df.id == id) & (cbonds_df.date == t0)].share))
        data_t1 = dict(zip(cbonds_df[(cbonds_df.id == id) & (cbonds_df.date == t1)].sec_isin, cbonds_df[(cbonds_df.id == id) & (cbonds_df.date == t1)].share))

        abs_sum_change = abs_sum(calculate_structure_difference(data_t0, data_t1))
        data_dict[t1] = abs_sum_change

    return data_dict


def foo(list_):
    id_ex_dict = {}
    for i in tqdm(list_):
        try:
            id_ex_dict[i] = change_share(i)
        except:
            print(i)
            continue
    return id_ex_dict


def max_data(data):
    return np.less_equal(data, pai_df.Дата.values)




def get_corr_all_sharp(index_fund, plot=False, use_rate=True):
    scaler = StandardScaler()
    # Получение имени фонда
    name_fund = cbonds_df.loc[cbonds_df.id == index_fund, 'name'].iloc[0]

    # Получение данных фонда
    pai_df['Дата'] = [pd.Timestamp(date) for date in pai_df['Дата']]
    # Получение данных фонда
    data_fund = set([pd.Timestamp(date) for date in result[index_fund].keys()]) & \
                set([pd.Timestamp(date) for date in pai_df.loc[pai_df['Фонд'] == name_fund, 'Дата']])


    # Подсчет разницы формы и нормализация
    diff_shape = np.array(list(result[index_fund].values())[1:])
    diff_shape = (diff_shape/100).reshape(1, -1)
    diff_shape_norm = scaler.fit_transform(diff_shape.reshape(-1, 1)).reshape(1, -1)
    diff_shape_mean = np.mean(diff_shape)


    # Подсчет коэффициента остроты и нормализация цены пая
    most_frequent_company_list = pai_df.loc[(pai_df['Фонд'] == name_fund) & pai_df['Дата'].isin(data_fund)]['УК'].value_counts().index

    for most_frequent_company in most_frequent_company_list:
        try:
            price = pai_df.loc[(pai_df['Фонд'] == name_fund) & (pai_df['УК'] == most_frequent_company) & pai_df['Дата'].isin(data_fund), 'Пай'].values
            price_diff = - np.diff(price) / price[:-1]
            if use_rate:
                rate = pai_df.loc[(pai_df['Фонд'] == name_fund) & (pai_df['УК'] == most_frequent_company) & pai_df['Дата'].isin(data_fund), 'rate'].values[1:]/100
                sharp = ((price_diff - rate) / np.std(price_diff - rate)).reshape(1, -1)
                target_norm = scaler.fit_transform(sharp.reshape(-1, 1)).reshape(1, -1)
                target_mean = np.mean(sharp)
                target = sharp
            else:
                profitability = price_diff/np.std(price_diff)
                target_norm = scaler.fit_transform(profitability.reshape(-1, 1)).reshape(1, -1)
                target_mean = np.mean(profitability)
                target = profitability
            
            # Подсчет корреляции
            r = cosine_similarity(diff_shape_norm, target_norm)            

            if plot:
                # Построение графика
                plt.scatter(diff_shape, sharp)
                plt.xlabel('Diff Shape Norm')
                plt.ylabel('Sharp Array Norm')
                plt.title('Correlation between Diff Shape Norm and Sharp Array Norm')
                plt.show()

            return [r[0][0], diff_shape_mean/2, target_mean, diff_shape, target]

        except:
            continue

    return(f'erorr_{index_fund}_{name_fund}')



def liner_reg(x_main, y_main, target='Sharpe', plot=False):

    mask = ((x_main <= 4) == (y_main >= -8)) == ((x_main >= -4) == (y_main <= 8))

    # Преобразование векторов в pandas DataFrame
    data = pd.DataFrame({'Volatility': x_main[mask], f'{target} Index': y_main[mask]})

    # Описательная статистика
    statistics = data.describe()
    print(statistics)

    # Корреляция
    correlation = data.corr()
    print(correlation)


    # Регрессионный анализ
    X = data['Volatility']
    y = data[f'{target} Index']
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm)
    results = model.fit()

    t_value = results.tvalues[1]
    p_value = results.pvalues[1]
    r_squared = results.rsquared
    std_error = np.sqrt(results.mse_resid)
    f_value = results.fvalue
    p_value_f = results.f_pvalue
    aic = results.aic
    bic = results.bic
    het_test = het_breuschpagan(results.resid, X_sm)
    autocorr_test = acorr_ljungbox(results.resid)

    print(f"Linear Regression (Volatility vs {target} Index)")
    print("t-value:", t_value)
    print("p-value:", p_value)
    print("R-squared:", r_squared)
    print("Standard Error of Regression:", std_error)
    print("F-value:", f_value)
    print("p-value (F-value):", p_value_f)
    print("AIC:", aic)
    print("BIC:", bic)
    print("Heteroscedasticity Test:", het_test)
    print("Autocorrelation Test:", autocorr_test)

    if plot:
        # Графики
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x='Volatility', y=f'{target} Index')
        plt.xlabel('Volatility')
        plt.ylabel(f'{target} Index')
        plt.title(f'Correlation between Volatility and {target} Index')
        plt.show()

        # Анализ выбросов
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data)
        plt.xlabel('Variables')
        plt.ylabel('Values')
        plt.title('Outlier Analysis')
        plt.show()

        # Группировка и сравнение
        # Если есть дополнительные категориальные переменные, можно провести группировку и сравнение
        # Например, с использованием seaborn.barplot или seaborn.boxplot

def poly_liner_reg(x_main, y_main, target='Sharpe', split_point = 0.2, plot=False):
    mask = ((x_main <= 4) == (y_main >= -8)) == ((x_main >= -4) == (y_main <= 8))

    # Преобразование векторов в pandas DataFrame
    data = pd.DataFrame({'Volatility': x_main[mask], 'Sharpe Index': y_main[mask]})

    # Разделение данных на участки
    # Точка разделения данных
    data1 = data[data['Volatility']<=split_point].reset_index(drop=True)
    data2 = data[data['Volatility']>split_point].reset_index(drop=True)

    # Нелинейная корреляция
    corr1 = np.corrcoef(data1['Volatility'], data1['Sharpe Index'])[0, 1]
    corr2 = np.corrcoef(data2['Volatility'], data2['Sharpe Index'])[0, 1]
    print("Correlation (Data1):", corr1)
    print("Correlation (Data2):", corr2)

    # Нелинейная регрессия
    X1 = data1['Volatility'].values.reshape(-1, 1)
    y1 = data1['Sharpe Index'].values
    X2 = data2['Volatility'].values.reshape(-1, 1)
    y2 = data2['Sharpe Index'].values
    X = data['Volatility'].values.reshape(-1, 1)
    y = data['Sharpe Index'].values

    # Построение полиномиальных признаков
    degree = 2  # Степень полинома (может быть изменена)
    poly_features = PolynomialFeatures(degree=1)
    X1_poly = poly_features.fit_transform(X1)
    X2_poly = poly_features.fit_transform(X2)

    X_poly = poly_features.fit_transform(X)


    # Обучение модели регрессии
    model1 = LinearRegression()
    model1.fit(X1_poly, y1)
    model2 = LinearRegression()
    model2.fit(X2_poly, y2)

    # Предсказание
    y1_pred = model1.predict(X1_poly)
    y2_pred = model2.predict(X2_poly)

    # Linear Regression
    print("Linear Regression (Data1)")

    X1_poly_sm = sm.add_constant(X1_poly)
    model1_sm = sm.OLS(y1, X1_poly_sm)
    results1 = model1_sm.fit()
    t_value1 = results1.tvalues[1]
    p_value1 = results1.pvalues[1]
    r_squared1 = results1.rsquared
    std_error1 = np.sqrt(results1.mse_resid)
    f_value1 = results1.fvalue
    p_value_f1 = results1.f_pvalue
    aic1 = results1.aic
    bic1 = results1.bic
    het_test1 = het_breuschpagan(results1.resid, X1_poly_sm)
    autocorr_test1 = acorr_ljungbox(results1.resid)

    print("t-статистика:", t_value1)
    print("p-значение:", p_value1)
    print("Коэффициент детерминации (R-squared):", r_squared1)
    print("Стандартная ошибка регрессии:", std_error1)
    print("F-статистика:", f_value1)
    print("p-значение (F-статистика):", p_value_f1)
    print("AIC:", aic1)
    print("BIC:", bic1)
    print("Тест на гетероскедастичность:", het_test1)
    print("Тест на автокорреляцию:", autocorr_test1)
    print('')
    print('')
    print('')
    # Linear Regression
    X2_poly_sm = sm.add_constant(X2_poly)
    model2_sm = sm.OLS(y2, X2_poly_sm)
    results2 = model2_sm.fit()
    t_value2 = results2.tvalues[1]
    p_value2 = results2.pvalues[1]
    r_squared2 = results2.rsquared
    std_error2 = np.sqrt(results2.mse_resid)
    f_value2 = results2.fvalue
    p_value_f2 = results2.f_pvalue
    aic2 = results2.aic
    bic2 = results2.bic
    het_test2= het_breuschpagan(results2.resid, X2_poly_sm)
    autocorr_test2 = acorr_ljungbox(results2.resid)

    print("Linear Regression (Data2)")
    print("t-статистика:", t_value2)
    print("p-значение:", p_value2)
    print("Коэффициент детерминации (R-squared):", r_squared2)
    print("Стандартная ошибка регрессии:", std_error2)
    print("F-статистика:", f_value2)
    print("p-значение (F-статистика):", p_value_f2)
    print("AIC:", aic2)
    print("BIC:", bic2)
    print("Тест на гетероскедастичность:", het_test2)
    print("Тест на автокорреляцию:", autocorr_test2)

    if plot:
        # Графики нелинейной регрессии
        plt.figure(figsize=(14, 8))
        plt.scatter(data1['Volatility'], data1['Sharpe Index'], label='Data1')
        plt.scatter(data2['Volatility'], data2['Sharpe Index'], label='Data2')
        plt.plot(data1['Volatility'], y1_pred, color='r', label='Regression (Data1)')
        plt.plot(data2['Volatility'], y2_pred, color='g', label='Regression (Data2)')
        plt.xlabel('Volatility')
        plt.ylabel('Sharpe Index')
        plt.title('Nonlinear Regression')
        plt.legend()
        plt.show()
