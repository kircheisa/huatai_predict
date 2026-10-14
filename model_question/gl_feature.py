# 遗传算法
from numpy.lib.stride_tricks import sliding_window_view
from gplearn.genetic import SymbolicTransformer

import numpy as np
import pandas as pd
from gplearn.functions import make_function
# from scipy.signal import find_peaks



# def autocorrelation(x, _):
#     """
#     计算时间序列的自相关系数。
#
#     参数:
#     x (array-like): 输入的时间序列数据。
#     _: 忽略的第二个参数（为了满足 arity=2 的要求）。
#
#     返回:
#     np.array: 与输入形状相同的数组，填充自相关系数。
#     """
#     if len(x) < 2:  # 数据长度不足以计算自相关
#         return np.full_like(x, 0.0)  # 返回与输入形状相同的数组，填充 0.0
#     acf = pd.Series(x).autocorr(lag=1)  # 计算滞后 1 的自相关系数
#     if np.isnan(acf):  # 如果自相关系数为 NaN，返回 0.0
#         return np.full_like(x, 0.0)
#     return np.full_like(x, acf)  # 返回与输入形状相同的数组，填充自相关系数
#
# # 使用 make_function 包装自定义函数
# autocorrelation_fn = make_function(
#     function=autocorrelation,  # 自定义函数
#     name='autocorrelation',  # 函数名称
#     arity=2  # 函数接受的参数数量
# )
#
#
# # 自定义函数：滚动相关性
# def rolling_correlation(x, y, window=5):
#     """
#     计算两个时间序列的滚动相关性。
#
#     参数:
#     x (array-like): 第一个时间序列数据。
#     y (array-like): 第二个时间序列数据。
#     window (int): 滑动窗口大小。
#
#     返回:
#     np.array: 滚动相关性结果。
#     """
#     if len(x) < window or len(y) < window:  # 数据长度不足以计算滚动相关性
#         return np.full_like(x, 0.0)  # 返回与输入形状相同的数组，填充 0.0
#     corr = pd.Series(x).rolling(window=window).corr(pd.Series(y)).values
#     return np.nan_to_num(corr, nan=0.0)  # 将 NaN 替换为 0.0
#
#
# # 包装滚动相关性函数，使其符合 arity=2 的要求
# def rolling_correlation_fn(x, y):
#     return rolling_correlation(x, y, window=5)
#
#
# # 使用 make_function 包装自定义函数
# rolling_correlation_fn = make_function(
#     function=rolling_correlation_fn,  # 自定义函数
#     name='rolling_correlation',  # 函数名称
#     arity=2  # 函数接受的参数数量
# )
#
#
# # 自定义函数：峰谷值检测
# def peak_trough(x):
#     """
#     检测时间序列中的峰值和谷值。
#
#     参数:
#     x (array-like): 输入的时间序列数据。
#
#     返回:
#     np.array: 峰值和谷值的标记数组（1为峰值，-1为谷值，0为其他）。
#     """
#     # 检测峰值
#     peaks, _ = find_peaks(x)
#     # 检测谷值（通过反转信号）
#     troughs, _ = find_peaks(-x)
#
#     # 创建标记数组
#     result = np.zeros_like(x)
#     result[peaks] = 1  # 标记峰值
#     result[troughs] = -1  # 标记谷值
#
#     return result
#
#
# # 使用 make_function 包装自定义函数
# peak_trough_fn = make_function(
#     function=peak_trough,  # 自定义函数
#     name='peak_trough',  # 函数名称
#     arity=1  # 函数接受的参数数量
# )
#
#
# # 自定义函数：峰谷位置检测
# def peak_trough_position(x):
#     """
#     检测时间序列中的峰值和谷值的位置。
#
#     参数:
#     x (array-like): 输入的时间序列数据。
#
#     返回:
#     np.array: 峰值和谷值的位置标记数组（1为峰值，-1为谷值，0为其他）。
#     """
#     # 检测峰值
#     peaks, _ = find_peaks(x)
#     # 检测谷值（通过反转信号）
#     troughs, _ = find_peaks(-x)
#
#     # 创建标记数组
#     result = np.zeros_like(x)
#     result[peaks] = 1  # 标记峰值
#     result[troughs] = -1  # 标记谷值
#
#     return result
#
#
# # 使用 make_function 包装自定义函数
# peak_trough_position_fn = make_function(
#     function=peak_trough_position,  # 自定义函数
#     name='peak_trough_position',  # 函数名称
#     arity=1  # 函数接受的参数数量
# )



# 感谢粉丝提供的函数集
def gen_function_set():

    windows = [5, 20, 60, 120]
    function_set = []
    for rolling_window in windows:
        def ts_delay(X):
            shifted = np.roll(X, rolling_window)
            return shifted


        function_name = f'ts_delay_{rolling_window}'
        globals()[function_name] = make_function(function=ts_delay, name=f'ts_delay_{rolling_window}', arity=1)
        function_set.append(function_name)


        def ts_corr(arr1, arr2):
            # print(rolling_window)
            if len(arr1) < rolling_window or len(arr2) < rolling_window:
                return np.zeros_like(arr1)
            else:
                window1 = sliding_window_view(arr1, window_shape=rolling_window)
                window2 = sliding_window_view(arr2, window_shape=rolling_window)

                mean1 = np.mean(window1, axis=1)
                mean2 = np.mean(window2, axis=1)
                std1 = np.std(window1, axis=1)
                std2 = np.std(window2, axis=1)

                cov = np.mean(window1 * window2, axis=1) - mean1 * mean2
                with np.errstate(divide='ignore', invalid='ignore'):
                    corr = cov / (std1 * std2)
                    corr[np.isnan(corr) | np.isinf(corr)] = 0
                return np.concatenate([np.zeros(rolling_window - 1), corr])


        function_name = f'ts_corr_{rolling_window}'
        globals()[function_name] = make_function(function=ts_corr, name=f'ts_corr_{rolling_window}', arity=2)
        function_set.append(function_name)


        def ts_cov(arr1, arr2):
            if len(arr1) < rolling_window or len(arr2) < rolling_window:
                return np.zeros_like(arr1)
            else:
                window1 = sliding_window_view(arr1, window_shape=rolling_window)
                window2 = sliding_window_view(arr2, window_shape=rolling_window)

                mean1 = np.mean(window1, axis=1)
                mean2 = np.mean(window2, axis=1)
                std1 = np.std(window1, axis=1)
                std2 = np.std(window2, axis=1)

                cov = np.mean(window1 * window2, axis=1) - mean1 * mean2
                cov[np.isnan(cov) | np.isinf(cov)] = 0
                return np.concatenate([np.zeros(rolling_window - 1), cov])


        function_name = f'ts_cov_{rolling_window}'
        globals()[function_name] = make_function(function=ts_cov, name=f'ts_cov_{rolling_window}', arity=2)
        function_set.append(function_name)


        def ts_max(arr):
            if len(arr) < rolling_window:
                return np.zeros_like(arr)
            else:
                rolling_max = np.max(sliding_window_view(arr, window_shape=rolling_window), axis=1)
                return np.concatenate([np.zeros(rolling_window - 1), rolling_max])


        function_name = f'ts_max_{rolling_window}'
        globals()[function_name] = make_function(function=ts_max, name=f'ts_max_{rolling_window}', arity=1)
        function_set.append(function_name)


        def ts_min(arr):
            if len(arr) < rolling_window:
                return np.zeros_like(arr)
            else:
                rolling_min = np.min(sliding_window_view(arr, window_shape=rolling_window), axis=1)
                return np.concatenate([np.zeros(rolling_window - 1), rolling_min])


        function_name = f'ts_min_{rolling_window}'
        globals()[function_name] = make_function(function=ts_min, name=f'ts_min_{rolling_window}', arity=1)
        function_set.append(function_name)


        def ts_maxarg(arr):
            if len(arr) < rolling_window:
                return np.zeros_like(arr)
            else:
                rolling_max_pos = np.argmax(sliding_window_view(arr, window_shape=rolling_window), axis=1)
                return np.concatenate([np.zeros(rolling_window - 1), rolling_max_pos])


        function_name = f'ts_maxarg_{rolling_window}'
        globals()[function_name] = make_function(function=ts_maxarg, name=f'ts_maxarg_{rolling_window}', arity=1)
        function_set.append(function_name)


        def ts_minarg(arr):
            if len(arr) < rolling_window:
                return np.zeros_like(arr)
            else:
                rolling_min_pos = np.argmin(sliding_window_view(arr, window_shape=rolling_window), axis=1)
                return np.concatenate([np.zeros(rolling_window - 1), rolling_min_pos])


        function_name = f'ts_minarg_{rolling_window}'
        globals()[function_name] = make_function(function=ts_minarg, name=f'ts_minarg_{rolling_window}', arity=1)
        function_set.append(function_name)


        def ts_mean(arr):
            if len(arr) < rolling_window:
                return np.zeros_like(arr)
            else:
                rolling_mean = np.mean(sliding_window_view(arr, window_shape=rolling_window), axis=1)
                return np.concatenate([np.zeros(rolling_window - 1), rolling_mean])


        function_name = f'ts_mean_{rolling_window}'
        globals()[function_name] = make_function(function=ts_mean, name=f'ts_mean_{rolling_window}', arity=1)
        function_set.append(function_name)


        def ts_median(arr):
            if len(arr) < rolling_window:
                return np.zeros_like(arr)
            else:
                rolling_median = np.median(sliding_window_view(arr, window_shape=rolling_window), axis=1)
                return np.concatenate([np.zeros(rolling_window - 1), rolling_median])


        function_name = f'ts_median_{rolling_window}'
        globals()[function_name] = make_function(function=ts_median, name=f'ts_median_{rolling_window}', arity=1)
        function_set.append(function_name)


        def ts_sum(arr):
            if len(arr) < rolling_window:
                return np.zeros_like(arr)
            else:
                rolling_sum = np.sum(sliding_window_view(arr, window_shape=rolling_window), axis=1)
                return np.concatenate([np.zeros(rolling_window - 1), rolling_sum])


        function_name = f'ts_sum_{rolling_window}'
        globals()[function_name] = make_function(function=ts_sum, name=f'ts_sum_{rolling_window}', arity=1)
        function_set.append(function_name)


        def ts_stddev(arr):
            if len(arr) < rolling_window:
                return np.zeros_like(arr)
            else:
                rolling_stddev = np.std(sliding_window_view(arr, window_shape=rolling_window), axis=1)
                return np.concatenate([np.zeros(rolling_window - 1), rolling_stddev])


        function_name = f'ts_stddev_{rolling_window}'
        globals()[function_name] = make_function(function=ts_stddev, name=f'ts_stddev_{rolling_window}', arity=1)
        function_set.append(function_name)


        def ts_product(arr):
            if len(arr) < rolling_window:
                return np.zeros_like(arr)
            else:
                rolling_product = np.prod(sliding_window_view(arr, window_shape=rolling_window), axis=1)
                return np.concatenate([np.zeros(rolling_window - 1), rolling_product])


        function_name = f'ts_product_{rolling_window}'
        globals()[function_name] = make_function(function=ts_product, name=f'ts_product_{rolling_window}', arity=1)
        function_set.append(function_name)


        def ts_kurt(arr):
            if len(arr) < rolling_window:
                return np.zeros_like(arr)
            else:
                windows = sliding_window_view(arr, window_shape=rolling_window)
                stds = np.std(windows, axis=1)
                means = np.mean(windows, axis=1, keepdims=True)
                deviations = windows - means
                mu4 = np.mean(deviations ** 4, axis=1)
                stds[stds == 0] = 1
                kurtosis_values = mu4 / stds ** 4 - 3
                return np.concatenate([np.zeros(rolling_window - 1), kurtosis_values])


        function_name = f'ts_kurt_{rolling_window}'
        globals()[function_name] = make_function(function=ts_kurt, name=f'ts_kurt_{rolling_window}', arity=1)
        function_set.append(function_name)


        def ts_skew(arr):
            if len(arr) < rolling_window:
                return np.zeros_like(arr)
            else:
                windows = sliding_window_view(arr, window_shape=rolling_window)
                stds = np.std(windows, axis=1)
                means = np.mean(windows, axis=1, keepdims=True)
                deviations = windows - means
                mu3 = np.mean(deviations ** 3, axis=1)
                stds[stds == 0] = 1
                skewness_values = mu3 / stds ** 3
                return np.concatenate([np.zeros(rolling_window - 1), skewness_values])


        function_name = f'ts_skew_{rolling_window}'
        globals()[function_name] = make_function(function=ts_skew, name=f'ts_skew_{rolling_window}', arity=1)
        function_set.append(function_name)


        def ts_zscore(arr):
            if len(arr) < rolling_window:
                return np.zeros_like(arr)
            else:
                windows = sliding_window_view(arr, window_shape=rolling_window)
                stds = np.std(windows, axis=1, keepdims=True)
                means = np.mean(windows, axis=1, keepdims=True)
                stds[stds == 0] = 1
                zscore_values = (windows - means) / stds
                zscore_values = zscore_values[:, -1]
                return np.concatenate([np.zeros(rolling_window - 1), zscore_values])


        function_name = f'ts_zscore_{rolling_window}'
        globals()[function_name] = make_function(function=ts_zscore, name=f'ts_zscore_{rolling_window}', arity=1)
        function_set.append(function_name)


        def decay_linear(arr):
            if len(arr) < rolling_window:
                return np.zeros_like(arr)
            else:
                windows = sliding_window_view(arr, window_shape=rolling_window)
                weights = np.arange(rolling_window, 0, -1)
                weights_normalized = weights / np.sum(weights)
                weighted_means = np.sum(windows * weights_normalized, axis=1)
                return np.concatenate([np.zeros(rolling_window - 1), weighted_means])


        function_name = f'decay_linear_{rolling_window}'
        globals()[function_name] = make_function(function=decay_linear, name=f'decay_linear_{rolling_window}', arity=1)
        function_set.append(function_name)


    windows = [20, 60, 120]
    N = [5, 15, 30]
    for i in range(len(windows)):
        rolling_window = windows[i]
        n = N[i]


        def ts_ascsortcut_(arr1):
            if len(arr1) < rolling_window:
                return np.zeros_like(arr1)
            else:
                partitioned = sliding_window_view(arr1, window_shape=rolling_window, axis=0)
                partitioned = np.partition(partitioned, n - 1, axis=-1)
                min_n = partitioned[..., :n]
                min_n = np.mean(min_n, axis=-1)
            return np.concatenate([np.zeros(rolling_window - 1), min_n])


        function_name = f'ts_ascsortcut_{rolling_window}_{n}'
        globals()[function_name] = make_function(function=ts_ascsortcut_, name=f'ts_ascsortcut_{rolling_window}_{n}',
                                                 arity=1)
        function_set.append(function_name)

    for i in range(len(windows)):
        rolling_window = windows[i]
        n = N[i]


        def ts_decsortcut(arr1):
            if len(arr1) < rolling_window:
                return np.zeros_like(arr1)
            else:
                partitioned = sliding_window_view(arr1, window_shape=rolling_window, axis=0)
                partitioned = np.partition(partitioned, -n, axis=-1)
                min_n = partitioned[..., -n:]
                min_n = np.mean(min_n, axis=-1)
            return np.concatenate([np.zeros(rolling_window - 1), min_n])


        function_name = f'ts_decsortcut{rolling_window}_{n}'
        globals()[function_name] = make_function(function=ts_decsortcut, name=f'ts_decsortcut_{rolling_window}_{n}',
                                                 arity=1)
        function_set.append(function_name)


    windows = [-5,-4,-3,-2,-1,0,1, 2, 3, 4, 5,]
    for i in range(len(windows)):
        def const(arr1):
            return arr1+windows[i]
        function_name = f'const_{windows[i]}'
        globals()[function_name] = make_function(function=const, name=f'const_{windows[i]}', arity=1)
        function_set.append(function_name)
    return function_set

def _logical1(x1, x2, x3, x4):
    return np.where(x1 > x2, x3, x4)
logical1 = make_function(function=_logical1,
                        name='logical1',
                        arity=4)

def _logical2(x1, x2):
    return np.where(x1 > x2, x1, x2)
logical2 = make_function(function=_logical2,
                        name='logical2',
                        arity=2)

def _logical3(x1, x2):
    return np.where(x1 > x2, 1, -1)
logical3 = make_function(function=_logical3,
                        name='logical3',
                        arity=2)

def _logical4(x1, x2):
    return np.where(x1 > x2, 1, 0)
logical4 = make_function(function=_logical4,
                        name='logical4',
                        arity=2)


function_set = [globals()[name] for name in gen_function_set()]
function_set += ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'abs', 'neg', 'inv', logical1, logical2, logical3, logical4]



# 创建 SymbolicTransformer
symbolic_transformer = SymbolicTransformer(
    population_size=100000,  # 种群大小
    generations=5,  # 进化代数
    function_set=function_set,  # 允许的函数
    n_components=10,
    metric='pearson',
    parsimony_coefficient=0.01,  # 简约系数，控制表达式的复杂度
    max_samples=0.9,  # 每次拟合时使用的样本比例
    tournament_size=20,  # 锦标赛选择的大小
    verbose=1,  # 打印进度
    random_state=42
)



file_name1 = "data/43_data.csv"
file_name2 = "data/43_rate.csv"
df1 = pd.read_csv(file_name1)
df2 = pd.read_csv(file_name2)
df2 = df2.iloc[:, 1:]
df = pd.merge(df1, df2, left_on="Unnamed: 0", right_on="time")

df["rate_delta_5d"] = df.loc[:, ["rate"]].shift(-5) - df.loc[:, ["rate"]]


def gen_label(x):
    if x < -0.03:
        return 0
    elif x < 0.03:
        return 1
    else:
        return 2


df["label"] = df["rate_delta_5d"].apply(gen_label)
df_label = df.loc[:,["rate", "label"]]
df = df.drop(columns=["time", "rate", "label"])

x_train = df
x_train = x_train.set_index("Unnamed: 0")
x_train = x_train.drop(columns=["date_list", "rate_delta_5d"])
x_train = x_train.fillna(99.00)
y_train = df_label.loc[:,"label"]

# 训练符号转换器并生成新特征
symbolic_transformer.fit(x_train, y_train)

# 生成新特征
x_train_new = symbolic_transformer.transform(x_train)
pd.DataFrame(x_train_new).to_csv("data/43_x.csv")
pd.DataFrame(y_train).to_csv("data/43_y.csv")
print("生成的特征示例（前 5 行）:")
print(x_train_new[-5:])
