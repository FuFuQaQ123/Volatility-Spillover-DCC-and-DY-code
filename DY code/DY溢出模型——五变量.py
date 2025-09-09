import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
from statsmodels.tools.sm_exceptions import InterpolationWarning
import warnings
from arch.unitroot import PhillipsPerron
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
from matplotlib.patches import Circle
from statsmodels.tsa.vector_ar import var_model
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test
from scipy import stats
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.graphics.gofplots import qqplot_2samples
import matplotlib as mpl
from matplotlib.patches import Rectangle


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'Arial Unicode MS']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 忽略 InterpolationWarning
warnings.filterwarnings('ignore', category=InterpolationWarning)

# 读取 Excel 文件
excel_file = pd.ExcelFile("../数据（原数据未对数收益率）.xlsx") # "../数据（原数据未对数收益率）.xlsx"
df = excel_file.parse('Data')  # 将 Excel 文件中的指定工作表解析为一个 DataFrame 对象。类似于SQL表

# 保存时间序列列，假设时间序列列名为 '时间'，请根据实际情况修改
time_series = df['时间']

# 测试
# print(df)

# 对 BDI、BDTI、BCTI、WTI、GPR列进行对数化处理  L3.1_修  新增BRENT替换WTI
columns_to_log = ['BDI', 'BDTI', 'BCTI', 'WTI', 'GPR']


# # 进行描述性统计
# descriptive_stats = df[columns_to_log].describe()
# print("描述性统计结果：")
# print(descriptive_stats)
#
# plt.rcParams.update({
#     # 'figure.dpi': 110,          # 核心：控制所有图像分辨率为110dpi
#     # 'savefig.dpi': 300,         # 保存图像时同样使用300dpi（避免保存后分辨率降低）
#     'font.size': 20,            # 全局字体大小（避免标题/标签字体不协调）
#     'font.sans-serif': ['Arial'],  # 统一字体（避免中文乱码，若需中文可换'WenQuanYi Zen Hei'）
#
# })
#
# # 绘制折线图
# plt.figure(figsize=(12, 8))
# for col in columns_to_log:
#     plt.plot(time_series, df[col], label=col)
#
# # 总体
# plt.xlabel('Time')
# plt.xticks(rotation=45)
# plt.ylabel('values')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
#
# # 分开绘制每一列的折线图
# for col in columns_to_log:
#     plt.figure(figsize=(10, 6))
#     plt.plot(time_series, df[col])
#     plt.title(f'{col}')
#     plt.xlabel('Time')
#     plt.xticks(rotation=45)
#     plt.ylabel('values')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

for col in columns_to_log:
    df[col] = np.where(df[col] > 0, np.log(df[col]), np.nan)  # 对数化处理
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)  # 替换 inf 为 NaN
    df[col] = df[col].ffill()  # 前向填充 NaN


df = df.dropna()
# 测试
# print(df)

# 第二步：计算对数收益率
log_returns = (df[columns_to_log].diff() * 100).dropna() # 数据放大 * 100
# df = df.dropna()  # 删除对数化处理后可能产生的 NaN
# df[columns_to_log] = log_returns  # 将对数收益率覆盖原始列  将对数收益率添加回原始 DataFrame

# 进行描述性统计
descriptive_stats = log_returns.describe()
print("描述性统计结果：")
print(descriptive_stats)

# 计算并输出对数收益率的中位数
log_returns_median = log_returns.median()
print("\n对数收益率的中位数：")
print(log_returns_median)

# 测试
# print(log_returns)


# ——————————————L2.3.1_修订 数据正态性检验——————————————
# 正态性检验函数
def normality_tests(data, column_name):
    """
    对数据进行多种正态性检验并返回结果
    """
    # 确保数据中没有缺失值
    data = data.dropna()

    # Shapiro-Wilk检验
    shapiro_stat, shapiro_p = stats.shapiro(data)

    # Anderson-Darling检验 (修正返回值处理)
    anderson_result = normal_ad(data)
    anderson_stat = anderson_result[0]
    anderson_pvalue = anderson_result[1]

    # 计算偏度和峰度
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)  # 超额峰度，正态分布为0

    return {
        'column': column_name,
        'shapiro_statistic': shapiro_stat,
        'shapiro_pvalue': shapiro_p,
        'anderson_statistic': anderson_stat,
        'anderson_pvalue': anderson_pvalue,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'sample_size': len(data)  # 添加样本量信息
    }


# 执行正态性检验并存储结果
normality_results = []
for col in columns_to_log:
    result = normality_tests(log_returns[col], col)
    normality_results.append(result)

# 将结果转换为DataFrame并显示
normality_df = pd.DataFrame(normality_results)
print("\n===== 正态性检验结果 =====")
print(normality_df.round(4))

# 解释正态性检验结果
print("\n===== 正态性检验结果解释 =====")
for _, row in normality_df.iterrows():
    print(f"\n{row['column']} (样本量: {row['sample_size']}):")
    # Shapiro-Wilk解释
    if row['shapiro_pvalue'] > 0.05:
        print(f"  Shapiro-Wilk检验: p值 = {row['shapiro_pvalue']:.4f} > 0.05，不能拒绝正态分布假设")
    else:
        print(f"  Shapiro-Wilk检验: p值 = {row['shapiro_pvalue']:.4f} ≤ 0.05，拒绝正态分布假设")

    # Anderson-Darling解释
    if row['anderson_pvalue'] > 0.05:
        print(f"  Anderson-Darling检验: p值 = {row['anderson_pvalue']:.4f} > 0.05，不能拒绝正态分布假设")
    else:
        print(f"  Anderson-Darling检验: p值 = {row['anderson_pvalue']:.4f} ≤ 0.05，拒绝正态分布假设")

    # 偏度解释
    if abs(row['skewness']) < 0.5:
        skewness_interpret = "近似对称"
    elif abs(row['skewness']) < 1:
        skewness_interpret = "轻度偏态"
    else:
        skewness_interpret = "显著偏态"
    print(f"  偏度 = {row['skewness']:.4f} ({skewness_interpret})")

    # 峰度解释
    if abs(row['kurtosis']) < 0.5:
        kurtosis_interpret = "近似正态峰度"
    elif abs(row['kurtosis']) < 1:
        kurtosis_interpret = "轻度偏离正态峰度"
    else:
        kurtosis_interpret = "显著偏离正态峰度"
    print(f"  超额峰度 = {row['kurtosis']:.4f} ({kurtosis_interpret})")

# ———————————————————————————————————————新版绘图————————————————————————
plt.rcParams.update({
    # 'figure.dpi': 110,          # 核心：控制所有图像分辨率为300dpi
    # 'savefig.dpi': 300,         # 保存图像时同样使用300dpi（避免保存后分辨率降低）
    'font.size': 20,            # 全局字体大小（避免标题/标签字体不协调）
    'font.sans-serif': ['Arial'],  # 统一字体（避免中文乱码，若需中文可换'WenQuanYi Zen Hei'）
})

# 可视化：直方图
plt.figure(figsize=(15, 12))
plt.subplots_adjust(
    top=0.92,        # 顶部边距
    hspace=0.3,      # 垂直方向子图间距
    wspace=0.4,      # 水平方向子图间距
    left=0.08,       # 左侧边距
    right=0.95       # 右侧边距
)

# 更深的颜色循环配置
colors = [
    ('#CC3333', '#990000'),  # 红色系（更深）
    ('#1A75FF', '#0047B3'),  # 蓝色系（更深）
    ('#33AA33', '#007700'),  # 绿色系（更深）
    ('#FF9900', '#CC7700'),  # 橙色系（更深）
    ('#CC3399', '#990066'),  # 粉色系（更深）
    ('#DDCC00', '#AA9900')   # 黄色系（更深）
]

for i, col in enumerate(columns_to_log, 1):
    ax = plt.subplot(2, 3, i)

    # 获取当前颜色对（取模操作确保颜色循环使用）
    hist_color, kde_color = colors[(i - 1) % len(colors)]
    # 获取数据并确保没有缺失值
    data = log_returns[col].dropna()

    # 1. 先绘制直方图
    sns.histplot(
        log_returns[col].dropna(),
        kde=True,  # 不在这里绘制KDE曲线
        color=hist_color,  # 直方图填充色
        edgecolor='white',  # 直方图柱形边框色
        linewidth=0.5  # 直方图边框宽度
    )


    plt.xlabel('Rate of return')

    # 获取子图位置
    bbox = ax.get_position()

    # 创建带边框的灰色标题块
    title_ax = plt.axes([bbox.x0, bbox.y0 + bbox.height, #  + 0.01
                         bbox.width, 0.04])
    # 添加带边框的矩形，edgecolor设置边框颜色，linewidth设置边框宽度
    title_ax.add_patch(plt.Rectangle(
        (0, 0), 1, 1,
        facecolor='lightgray',  # 灰色背景
        edgecolor='black',  # 边框颜色（深灰色，与背景形成对比）
        linewidth=1.2,  # 边框宽度
        alpha=0.7  # 透明度
    ))
    title_ax.text(0.5, 0.5, f'{col} Yield distribution',
                  ha='center', va='center', fontsize=12, transform=title_ax.transAxes)
    title_ax.axis('off')



# Q-Q图
plt.figure(figsize=(15, 12))
plt.subplots_adjust(
    top=0.92,        # 顶部边距
    hspace=0.3,      # 垂直方向子图间距
    wspace=0.4,      # 水平方向子图间距
    left=0.08,       # 左侧边距
    right=0.95       # 右侧边距
)

for i, col in enumerate(columns_to_log, 1):
    ax = plt.subplot(2, 3, i)
    stats.probplot(log_returns[col].dropna(), plot=plt)
    plt.title('')

    # 获取子图位置
    bbox = ax.get_position()

    # 创建带边框的灰色标题块
    title_ax = plt.axes([bbox.x0, bbox.y0 + bbox.height,  # + 0.01
                         bbox.width, 0.04])
    title_ax.add_patch(plt.Rectangle(
        (0, 0), 1, 1,
        facecolor='lightgray',
        edgecolor='black',  # 边框颜色
        linewidth=1.2,  # 边框宽度
        alpha=0.7
    ))
    title_ax.text(0.5, 0.5, f'{col} Q-Q plot',
                  ha='center', va='center', fontsize=12, transform=title_ax.transAxes)
    title_ax.axis('off')


plt.show()
# ———————————————————————————————————————新版绘图————————————————————————


# 可视化：直方图和Q-Q图 ——————————————————旧版绘图————————————————————————
plt.figure(figsize=(15, 10))
for i, col in enumerate(columns_to_log, 1):
    # 直方图
    plt.subplot(2, 3, i)
    sns.histplot(log_returns[col].dropna(), kde=True)
    plt.title(f'{col} Yield distribution') # 收益率分布
    plt.xlabel('Rate of return') # 收益率



# Q-Q图
plt.figure(figsize=(15, 10))
for i, col in enumerate(columns_to_log, 1):
    plt.subplot(2, 3, i)
    stats.probplot(log_returns[col].dropna(), plot=plt)
    plt.title(f'{col} Q-Q plot')

plt.tight_layout()
plt.show()
# ——————————————L2.3.1_修订 数据正态性检验——————————————


# ADF 检验函数
def adf_test(column_data, column_name):
    if np.all(column_data == column_data.iloc[0]):
        print(f"Column {column_name} is a constant sequence. Skipping ADF test.")
        return None

    """
    result 是一个包含检验结果的元组，通常包含以下内容：
    检验统计量（ADF Statistic）
    p 值
    使用的滞后阶数
    用于检验的样本数量
    不同置信水平下的临界值（例如 1%，5%，10%）
    """
    result = adfuller(column_data)
    adf_statistic = result[0]
    critical_value_1_percent = result[4]['1%']
    return [column_name, adf_statistic, critical_value_1_percent]


# PP 检验函数
def pp_test(column_data, column_name):
    result = PhillipsPerron(column_data)
    pp_statistic = result.stat
    critical_value_1_percent = result.critical_values['1%']
    return [column_name, pp_statistic, critical_value_1_percent] # 时间序列数据的名称。PP 检验统计量。1% 置信水平下的临界值。


# 存储 ADF 和 PP 的列表
adf_results = []
pp_results = []

# 对各列进行 ADF 和 PP 检验并存储结果
for col in columns_to_log:
    adf_result = adf_test(log_returns[col], col)
    if adf_result:
        adf_results.append(adf_result)

    pp_result = pp_test(log_returns[col], col)
    pp_results.append(pp_result)

# 输出 ADF 检验结果  ——————————————L2.1_修订，VAR稳定性检验——————————————
print("ADF 检验结果:")
adf_df = pd.DataFrame(adf_results, columns=['列名', 'ADF 值', '1% 显著性水平临界值'])
print(adf_df)

# 输出 PP 检验结果
print("\nPP 检验结果:")
pp_df = pd.DataFrame(pp_results, columns=['列名', 'PP 值', '1% 显著性水平临界值'])
print(pp_df)


        # epsilon = 1e-10
        # # 2. 数据预处理：计算对数收益率
        # log_returns = (df[columns_to_log].clip(lower=epsilon) - df[columns_to_log].shift(1).clip(lower=epsilon)).dropna()

# 同时对时间序列列进行相同的操作，保证索引一致
time_series = time_series[log_returns.index]

# 将时间序列列设置为 log_returns 的索引
log_returns.index = time_series[log_returns.index]

        # 3. 平稳性检验（ADF 检验）（表 1）
        # print("\n对数收益率的 ADF 检验结果:")
        # for col in log_returns.columns:
        #     if not np.all(log_returns[col] == log_returns[col].iloc[0]):
        #         result = adfuller(log_returns[col])
        #         print(f"{col}: ADF Statistic = {result[0]}, p-value = {result[1]}")

# 确保索引单调
log_returns = log_returns.sort_index()

# print(log_returns)

# 确保索引是 DatetimeIndex 或 PeriodIndex 并设置频率
if not isinstance(log_returns.index, pd.DatetimeIndex):
    log_returns.index = pd.to_datetime(log_returns.index)

# 使用 to_period() 方法设置频率
log_returns.index = log_returns.index.to_period('M')

# print(log_returns)

# 4. 构建 VAR 模型
model = VAR(log_returns)
selected_lags = model.select_order(maxlags=20)
# print(selected_lags.summary())
"""
这是一个方法，返回一个摘要报告，显示不同滞后期数下的信息准则值。
输出结果通常包括：
AIC（赤池信息准则）：越小越好。
BIC（贝叶斯信息准则）：越小越好。
FPE（最终预测误差）：越小越好。
HQIC（汉南-昆信息准则）：越小越好。
通常选择 AIC 或 BIC 或 HQIC 最小的滞后期数作为最佳滞后期数。
"""
best_lags = selected_lags.aic # 提取基于 AIC 准则选择的最佳滞后期数。
results = model.fit(maxlags=best_lags)

print(f"滞后阶数p = {best_lags}") # L1.1_修订  VAR滞后阶数p，通过AIC准则自动计算最佳滞后阶数，P = 6

# p1 = selected_lags.aic
# p2 = selected_lags.bic
# p3 = selected_lags.hqic
# print(p1)
# print(p2)
# print(p3)


"""
拟合完成后，results 对象提供了多种方法和属性，用于分析模型结果。以下是一些常用的方法：
(1) 查看模型摘要:          print(results.summary())
(2) 脉冲响应分析（IRF）:   irf = results.irf(5)  # 计算 5 期的脉冲响应
                        irf.plot()
(3) 方差分解（FEVD）      fevd = results.fevd(5)  # 计算 5 期的方差分解,分析每个变量对系统方差的贡献
                        fevd.plot() # 会生成方差分解的图表。
(4) 预测                  forecast = results.forecast(log_returns.values[-best_lags:], steps=5)

print(best_lags)  # 6
print(results.summary())
"""
# print(results.summary())
# ——————————————L2.3.1_修订 VAR模型残差正态性检验——————————————
def var_residual_normality_test(residuals, var_names):
    """
    VAR模型残差正态性检验（多变量）
    参数：
        residuals: VAR模型残差（DataFrame，每行一个观测，每列一个变量的残差）
        var_names: 变量名称列表（与残差列对应）
    返回：
        normality_df: 检验结果汇总表
    """
    normality_results = []

    for var_name in var_names:
        res_data = residuals[var_name].dropna()  # 提取单个变量的残差（排除缺失值）
        n = len(res_data)

        # 1. Shapiro-Wilk检验（小样本更有效，n<5000）
        if n >= 3 and n <= 5000:  # Shapiro-Wilk对样本量有要求（3<=n<=5000）
            shapiro_stat, shapiro_p = stats.shapiro(res_data)
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan  # 超出范围时标记为NaN

        # 2. Anderson-Darling检验（适合大样本，对偏离正态更敏感）
        ad_stat, ad_p = normal_ad(res_data)  # statsmodels的修正AD检验

        # 3. 矩特征（偏度、超额峰度）
        skewness = stats.skew(res_data)  # 对称分布偏度≈0；正偏>0，负偏<0
        excess_kurtosis = stats.kurtosis(res_data)  # 正态分布超额峰度=0；尖峰>0，平峰<0

        # 4. Jarque-Bera检验（基于偏度和峰度的大样本检验）
        jb_stat, jb_p = stats.jarque_bera(res_data)

        # 存储单变量检验结果
        normality_results.append({
            '变量名': var_name,
            '样本量': n,
            'Shapiro-Wilk统计量': round(shapiro_stat, 4) if not np.isnan(shapiro_stat) else 'N/A',
            'Shapiro-Wilk p值': round(shapiro_p, 4) if not np.isnan(shapiro_p) else 'N/A',
            'Anderson-Darling统计量': round(ad_stat, 4),
            'Anderson-Darling p值': round(ad_p, 4),
            'Jarque-Bera统计量': round(jb_stat, 4),
            'Jarque-Bera p值': round(jb_p, 4),
            '偏度': round(skewness, 4),
            '超额峰度': round(excess_kurtosis, 4)
        })

    # 转换为DataFrame便于查看
    normality_df = pd.DataFrame(normality_results)
    return normality_df


# 1. 提取VAR模型残差
var_residuals = results.resid  # DataFrame格式：行=时间，列=变量残差
print("\n" + "=" * 50)
print("4. VAR模型残差基本信息")
print("=" * 50)
print(f"残差维度：{var_residuals.shape}（时间观测数 × 变量数）")
print(f"残差缺失值情况：\n{var_residuals.isnull().sum()}")

# 2. 执行残差正态性检验
normality_result_df = var_residual_normality_test(var_residuals, columns_to_log)
print("\n" + "=" * 50)
print("5. VAR残差正态性检验结果汇总")
print("=" * 50)
print(normality_result_df)

# 3. 自动化结果解释（基于0.05显著性水平）
print("\n" + "=" * 50)
print("6. 残差正态性检验结果解释（显著性水平 α=0.05）")
print("=" * 50)
for _, row in normality_result_df.iterrows():
    print(f"\n【{row['变量名']}】（样本量：{row['样本量']}）：")

    # Shapiro-Wilk解释
    if row['Shapiro-Wilk p值'] != 'N/A':
        if row['Shapiro-Wilk p值'] > 0.05:
            sw_interpret = "不能拒绝正态分布假设（残差近似正态）"
        else:
            sw_interpret = "拒绝正态分布假设（残差显著偏离正态）"
        print(f"  - Shapiro-Wilk检验：{sw_interpret}（p值={row['Shapiro-Wilk p值']}）")
    else:
        print(f"  - Shapiro-Wilk检验：样本量超出适用范围（3≤n≤5000），结果无效")

    # Anderson-Darling解释
    if row['Anderson-Darling p值'] > 0.05:
        ad_interpret = "不能拒绝正态分布假设（残差近似正态）"
    else:
        ad_interpret = "拒绝正态分布假设（残差显著偏离正态）"
    print(f"  - Anderson-Darling检验：{ad_interpret}（p值={row['Anderson-Darling p值']}）")

    # Jarque-Bera解释
    if row['Jarque-Bera p值'] > 0.05:
        jb_interpret = "不能拒绝正态分布假设（残差近似正态）"
    else:
        jb_interpret = "拒绝正态分布假设（残差显著偏离正态）"
    print(f"  - Jarque-Bera检验：{jb_interpret}（p值={row['Jarque-Bera p值']}）")

    # 矩特征解释
    # 偏度解释
    if abs(row['偏度']) < 0.5:
        skew_interpret = "近似对称"
    elif abs(row['偏度']) < 1:
        skew_interpret = "轻度偏态"
    else:
        skew_interpret = "显著偏态"
    print(f"  - 偏度：{row['偏度']}（{skew_interpret}）")

    # 超额峰度解释
    if abs(row['超额峰度']) < 0.5:
        kurt_interpret = "近似正态峰度（ mesokurtic ）"
    elif abs(row['超额峰度']) < 1:
        kurt_interpret = "轻度尖峰/平峰"
    else:
        kurt_interpret = "显著尖峰（ leptokurtic ）/平峰（ platykurtic ）"
    print(f"  - 超额峰度：{row['超额峰度']}（{kurt_interpret}）")

# ---------------------- 新增：残差正态性可视化 ----------------------
print("\n" + "=" * 50)
print("7. 残差正态性可视化（直方图+Q-Q图+残差时序图）")
print("=" * 50)
# ---------------------- 子图1：残差直方图 + 理论正态曲线（5个变量，2行3列布局） ----------------------

# 设置全局样式
plt.rcParams.update({
    'font.size': 20,  # 全局字体大小
    'font.sans-serif': ['Arial'],  # 统一字体
})

# 定义更深的颜色方案，与之前的直方图风格保持一致
colors = [
    ('#CC3333', '#990000'),  # 红色系
    ('#1A75FF', '#0047B3'),  # 蓝色系
    ('#33AA33', '#007700'),  # 绿色系
    ('#FF9900', '#CC7700'),  # 橙色系
    ('#CC3399', '#990066')  # 粉色系
]

plt.figure(figsize=(15, 12))  # 增加高度以容纳标题块
plt.subplots_adjust(
    top=0.92,  # 调整顶部边距，为标题块留出空间
    hspace=0.3,  # 垂直方向子图间距
    wspace=0.4,  # 水平方向子图间距
    left=0.08,  # 左侧边距
    right=0.95,  # 右侧边距
    bottom=0.15
)

for i, var_name in enumerate(columns_to_log):
    res_data = var_residuals[var_name].dropna()
    # 创建子图（2行3列，第i+1个位置）
    ax = plt.subplot(2, 3, i + 1)

    # 获取对应颜色
    hist_color, line_color = colors[i % len(colors)]

    # 绘制残差直方图（使用统一颜色方案）
    plt.hist(
        res_data,
        bins=20,
        density=True,
        alpha=0.6,
        color=hist_color,  # 使用统一颜色
        edgecolor='white',  # 白色边框，更清晰
        linewidth=0.8,
        label='Residual Histogram'
    )

    # 生成理论正态分布曲线
    x_norm = np.linspace(res_data.min(), res_data.max(), 100)
    y_norm = stats.norm.pdf(x_norm, loc=res_data.mean(), scale=res_data.std())
    plt.plot(
        x_norm,
        y_norm,
        color=line_color,  # 使用对应深色作为线色
        linewidth=2,
        label='Theoretical Normal'
    )

    # 子图标签与格式
    plt.xlabel('Residual Value', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.legend(fontsize=11, loc='upper right')


    # 获取子图位置，创建灰色标题块
    bbox = ax.get_position()
    title_ax = plt.axes([
        bbox.x0,
        bbox.y0 + bbox.height,  # 子图上方
        bbox.width,
        0.04  # 标题块高度
    ])

    # 添加带边框的灰色背景
    title_ax.add_patch(plt.Rectangle(
        (0, 0), 1, 1,
        facecolor='lightgray',
        edgecolor='black',
        linewidth=1.2,
        alpha=0.7
    ))

    # 添加标题文本
    title_ax.text(
        0.5, 0.5,
        f'{var_name} Residual Distribution',
        ha='center',
        va='center',
        fontsize=14,
        transform=title_ax.transAxes
    )
    title_ax.axis('off')  # 隐藏标题轴

plt.show()

# plt.figure(figsize=(24, 7))  # 宽20、高4，适配5个并列子图
# for i, var_name in enumerate(columns_to_log):
#     res_data = var_residuals[var_name].dropna()
#     # 创建子图（1行5列，第i+1个位置）
#     plt.subplot(1, 5, i+1)
#     # 绘制残差直方图（密度模式，便于与理论分布对比）
#     plt.hist(res_data, bins=20, density=True, alpha=0.6, color=f'C{i}', label='Residual Histogram') # 残差直方图
#     # 生成理论正态分布曲线（匹配残差的均值和标准差）
#     x_norm = np.linspace(res_data.min(), res_data.max(), 100)  # 正态分布x轴范围
#     y_norm = stats.norm.pdf(x_norm, loc=res_data.mean(), scale=res_data.std())  # 正态分布概率密度
#     plt.plot(x_norm, y_norm, 'r-', linewidth=2, label='Theoretical Normal Distribution') # 理论正态分布
#     # 子图标签与格式
#     plt.title(f'{var_name} Residual Distribution', fontsize=11) # 残差分布
#     plt.xlabel(' Residual Value', fontsize=9) # 残差值
#     plt.ylabel('Density', fontsize=9) # 密度
#     plt.legend(fontsize=8, loc='upper right')
#     plt.grid(alpha=0.3)
#
#
# plt.tight_layout()  # 自动调整子图间距，避免标签重叠
# plt.show()

# ---------------------- 子图2：残差Q-Q图（对比正态分布，2行3列布局） ----------------------

plt.rcParams.update({
    'font.size': 20,  # 全局字体大小
    'font.sans-serif': ['Arial'],  # 统一字体
})

# 调整图表大小和布局，为标题块留出空间
plt.figure(figsize=(15, 12))  # 增加高度以容纳标题块
plt.subplots_adjust(
    top=0.92,  # 调整顶部边距
    hspace=0.3,  # 垂直方向子图间距
    wspace=0.4,  # 水平方向子图间距
    left=0.08,  # 左侧边距
    right=0.95,  # 右侧边距
    bottom=0.15  # 底部边距
)

for i, var_name in enumerate(columns_to_log):
    res_data = var_residuals[var_name].dropna()
    # 创建子图并获取轴对象
    ax = plt.subplot(2, 3, i + 1)

    # 绘制Q-Q图（对比标准正态分布）
    stats.probplot(res_data, dist="norm", plot=plt, fit=True)

    # 清除默认标题，使用自定义标题块
    plt.title('')

    # 子图标签与格式
    plt.xlabel('Theoretical Quantile', fontsize=18)  # 理论分位数
    plt.ylabel('Residual Empirical Quantile', fontsize=18)  # 残差实际分位数


    # 获取子图位置信息
    bbox = ax.get_position()

    # 创建带边框的灰色标题块
    title_ax = plt.axes([
        bbox.x0,
        bbox.y0 + bbox.height,  # 位于子图上方
        bbox.width,
        0.04  # 标题块高度
    ])

    # 添加灰色背景块
    title_ax.add_patch(plt.Rectangle(
        (0, 0), 1, 1,
        facecolor='lightgray',
        edgecolor='black',  # 边框颜色
        linewidth=1.2,  # 边框宽度
        alpha=0.7  # 透明度
    ))

    # 添加标题文本
    title_ax.text(
        0.5, 0.5,
        f'{var_name} Residual Q-Q Plot',
        ha='center',
        va='center',
        fontsize=12,
        transform=title_ax.transAxes
    )

    # 隐藏标题轴
    title_ax.axis('off')


plt.show()

# plt.figure(figsize=(20, 4))
# for i, var_name in enumerate(columns_to_log):
#     res_data = var_residuals[var_name].dropna()
#     plt.subplot(1, 5, i+1)
#     # 绘制Q-Q图（dist="norm"表示对比标准正态分布）
#     stats.probplot(res_data, dist="norm", plot=plt, fit=True)
#     # 子图标签与格式
#     plt.title(f'{var_name} Residual Q-Q Plot', fontsize=11) # 残差Q-Q图
#     plt.xlabel('Theoretical Quantile', fontsize=9) # 理论分位数
#     plt.ylabel('Residual Empirical Quantile', fontsize=9) # 残差实际分位数
#     plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.show()
#
# ---------------------- 子图3：残差时序图（检查波动聚类，2行3列布局，适配5个变量） ----------------------
#
# plt.rcParams.update({
#     # 'figure.dpi': 110,          # 核心：控制所有图像分辨率为300dpi
#     # 'savefig.dpi': 300,         # 保存图像时同样使用300dpi（避免保存后分辨率降低）
#     'font.size': 20,            # 全局字体大小（避免标题/标签字体不协调）
#     'font.sans-serif': ['Arial'],  # 统一字体（避免中文乱码，若需中文可换'WenQuanYi Zen Hei'）
# })
#
# plt.figure(figsize=(18, 10))
# for i, var_name in enumerate(columns_to_log):
#     # 2行3列布局，i从0到4，对应子图位置1到5
#     plt.subplot(2, 3, i+1)
#     # 绘制残差时序（x轴为时间，y轴为残差值）
#     plt.plot(var_residuals.index.astype(str), var_residuals[var_name], color=f'C{i}', alpha=0.7)
#     plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Residual Mean（0）') # 残差均值
#     # 子图标签与格式（x轴时间标签旋转45度，避免重叠）
#     plt.title(f'{var_name} Residual time series', fontsize=11) # 残差时序
#     plt.xlabel('time', fontsize=9) # time
#     plt.ylabel('Residual value', fontsize=9) # 残差值
#     plt.xticks(rotation=45, fontsize=8)  # 旋转x轴标签，提升可读性
#     plt.legend(fontsize=8, loc='upper right')
#     plt.grid(alpha=0.3)
# # 删除第6个空白子图（2行3列共6个位置，仅用前5个）
# plt.delaxes(plt.gcf().axes[-1])
# plt.tight_layout()
# plt.show()
#
# # ---------------------- 额外：残差平方时序图（直观展示波动聚类） ----------------------
#
# plt.rcParams.update({
#     # 'figure.dpi': 110,          # 核心：控制所有图像分辨率为300dpi
#     # 'savefig.dpi': 300,         # 保存图像时同样使用300dpi（避免保存后分辨率降低）
#     'font.size': 20,            # 全局字体大小（避免标题/标签字体不协调）
#     'font.sans-serif': ['Arial'],  # 统一字体（避免中文乱码，若需中文可换'WenQuanYi Zen Hei'）
# })
#
# print("\n" + "="*50)
# print("8. 残差平方时序图（验证波动聚类）")
# print("="*50)
# plt.figure(figsize=(18, 10))
# for i, var_name in enumerate(columns_to_log):
#     plt.subplot(2, 3, i+1)
#     # 残差平方 = 波动 proxy（平方越大，波动越剧烈）
#     residual_sq = var_residuals[var_name] ** 2
#     plt.plot(var_residuals.index.astype(str), residual_sq, color=f'C{i}', alpha=0.7)
#     # 子图标签与格式
#     plt.title(f'{var_name} Residual squared time series fluctuation', fontsize=11) # 残差平方时序（波动 proxy）
#     plt.xlabel('time', fontsize=9) # 时间
#     plt.ylabel('Residual sum of squares', fontsize=9) # 残差平方值
#     plt.xticks(rotation=45, fontsize=8)
#     plt.grid(alpha=0.3)
# # 删除空白子图
# plt.delaxes(plt.gcf().axes[-1])
# plt.tight_layout()
# plt.show()

# ——————————————L2.3.1_修订 VAR模型残差正态性检验——————————————


# ——————————————L2.2.1_修订 全局VAR模型残差自相关检验（Ljung-Box）——————————————  ~~已完成~~
# 获取VAR模型的残差
residuals = results.resid

# 对残差进行Ljung-Box检验（检验是否为白噪声）
# 对于多变量时间序列，需要对每个变量的残差分别检验
for col in residuals.columns:
    lb_result = lb_test(residuals[col], lags=best_lags)  # 通常lags取模型滞后阶数或稍大
    print(f"\n{col}的Ljung-Box检验结果:")
    print("滞后阶数 | 统计量 | p值")
    print(lb_result)
# ——————————————L2.2.1_修订 全局VAR模型残差自相关检验（Ljung-Box）——————————————

# ——————————————L2.1.2_修订 进行格林兰因果检验——————————————
# 1. 定义变量列表（与columns_to_log一致，确保顺序）
variables = columns_to_log  # ['BDI', 'BDTI', 'BCTI', 'WTI', 'GPR']

# 2. 初始化结果存储列表（记录：原假设、滞后阶数、p值、是否拒绝原假设）
granger_results = []

# 3. 遍历所有变量对，进行双向格兰杰检验
for target in variables:  # target：被解释变量（如"BDTI"）
    for cause in variables:  # cause：解释变量（如"BDI"）
        if target != cause:  # 排除"变量自身对自身的检验"（无意义）
            # 准备检验数据：[被解释变量, 解释变量]（grangercausalitytests要求的顺序）
            test_data = log_returns[[target, cause]].dropna()  # 使用对数收益率数据，并确保无缺失值

            # 执行格兰杰检验：maxlag=best_lags（仅检验最优滞后阶数，避免冗余）
            # verbose=False：不输出详细中间结果，仅通过返回值提取关键信息
            result = grangercausalitytests(
                x=test_data,
                maxlag=best_lags,
                addconst=True,  # 回归中加入常数项（默认True，符合VAR模型设定）
                # verbose=False  新版本已弃用该参数
            )

            # 提取该滞后阶数的检验结果（result的key为滞后阶数，取best_lags对应的结果）
            lag_result = result[best_lags]

            # 提取关键统计量：F检验的p值（常用且直观，也可选择卡方检验p值）
            # lag_result[0]是字典，包含'lrtest'（卡方）、'params_ftest'（F检验）等
            f_statistic = lag_result[0]['params_ftest'][0]
            p_value = lag_result[0]['params_ftest'][1]

            # 判断是否拒绝原假设（原假设："cause不是target的格兰杰原因"）
            alpha = 0.05  # 显著性水平（常用0.05）
            reject_null = p_value < alpha
            conclusion = "拒绝原假设（存在格兰杰因果关系）" if reject_null else "接受原假设（无格兰杰因果关系）"

            # 将结果存入列表
            granger_results.append({
                "被解释变量(Target)": target,
                "解释变量(Cause)": cause,
                "滞后阶数(Lag)": best_lags,
                "F统计量": round(f_statistic, 4),
                "p值": round(p_value, 4),
                "结论": conclusion
            })

# 4. 整理结果为DataFrame并打印
granger_df = pd.DataFrame(granger_results)
print("\n所有变量对的格兰杰因果检验结果：")
print(granger_df.to_string(index=False))  # 不显示行索引，更清晰

# 5. （可选）筛选显著的因果关系（p<0.05）
significant_granger = granger_df[granger_df["p值"] < 0.05]
print(f"\n显著的格兰杰因果关系（p<0.05）共{len(significant_granger)}组：")
if len(significant_granger) > 0:
    print(significant_granger.to_string(index=False))
else:
    print("无显著的格兰杰因果关系（所有p值≥0.05）")
# ——————————————L2.1.2_修订 进行格林兰因果检验——————————————

# ——————————————L2.1.3_修订 验证模型的稳定性——————————————
# model_test = VAR(log_returns)
# fitted_model = model_test.fit(maxlags=best_lags)
stability = results.is_stable()
print('模型是否稳定:')
print(stability)
# if stability:
#     print("模型稳定")
# else:
#     print("模型不稳定")
# ——————————————L2.1.3_修订 验证模型的稳定性——————————————




# 动态计算
# 滚动窗口大小
rolling_window = 200  # L1.3_修订  时变溢出的滚动窗口长度为45

# 初始化存储溢出指数的列表
bdi_to_bdi = []
bdi_to_bdti = []
bdi_to_bcti = []
bdi_to_wti = []
bdi_to_gpr = []

bdti_to_bdi = []
bdti_to_bdti = []
bdti_to_bcti = []
bdti_to_wti = []
bdti_to_gpr = []

bcti_to_bdi = []
bcti_to_bdti = []
bcti_to_bcti = []
bcti_to_wti = []
bcti_to_gpr = []

wti_to_bdi = []
wti_to_bdti = []
wti_to_bcti = []
wti_to_wti = []
wti_to_gpr = []

gpr_to_bdi = []
gpr_to_bdti = []
gpr_to_bcti = []
gpr_to_wti = []
gpr_to_gpr = []

# 初始化存储总溢出指数的列表
total_spillover = []

window_start_times = []  # 用于存储每个滚动窗口的起始时间

window_p = [] # 用于存储每个滚动窗口自动拟合最佳的滞后阶数p

# 对对数收益率数据 log_returns 进行滚动窗口分析
for i in range(len(log_returns) - rolling_window):
    # 提取滚动窗口数据

    window_data = log_returns.iloc[i:i + rolling_window]

    # 拟合 VAR 模型
    model = VAR(window_data)

    # 静态：
    window_model = model.fit(maxlags=selected_lags.aic)
    window_p.append(selected_lags.aic)
    # 动态：
    # window_selected_lags = model.select_order(maxlags=20)  # 与全局一致，maxlags=6
    # window_best_lags = window_selected_lags.aic  # 窗口内AIC最优滞后阶数，结果均是6
    # window_model = model.fit(maxlags=window_best_lags)
    # window_p.append(window_best_lags)


    # 计算 FEVD
    fevd = window_model.fevd(20) # L1.2_修订  FEVD的预测水平H，手动设定

    try:
        # 提取最后一个预测步长的方差分解矩阵（第二维）
        fevd_decomp = fevd.decomp[:, -1, :]

        # 获取变量索引
        bdi_index = list(log_returns.columns).index('BDI')
        bdti_index = list(log_returns.columns).index('BDTI')
        bcti_index = list(log_returns.columns).index('BCTI')
        wti_index = list(log_returns.columns).index('WTI') # WTI 与  BRENT 互换
        gpr_index = list(log_returns.columns).index('GPR')

        # 提取溢出指数
        bdi_to_bdi.append(fevd_decomp[bdi_index][bdi_index])
        bdi_to_bdti.append(fevd_decomp[bdi_index][bdti_index])
        bdi_to_bcti.append(fevd_decomp[bdi_index][bcti_index])
        bdi_to_wti.append(fevd_decomp[bdi_index][wti_index])
        bdi_to_gpr.append(fevd_decomp[bdi_index][gpr_index])

        bdti_to_bdi.append(fevd_decomp[bdti_index][bdi_index])
        bdti_to_bdti.append(fevd_decomp[bdti_index][bdti_index])
        bdti_to_bcti.append(fevd_decomp[bdti_index][bcti_index])
        bdti_to_wti.append(fevd_decomp[bdti_index][wti_index])
        bdti_to_gpr.append(fevd_decomp[bdti_index][gpr_index])

        bcti_to_bdi.append(fevd_decomp[bcti_index][bdi_index])
        bcti_to_bdti.append(fevd_decomp[bcti_index][bdti_index])
        bcti_to_bcti.append(fevd_decomp[bcti_index][bcti_index])
        bcti_to_wti.append(fevd_decomp[bcti_index][wti_index])
        bcti_to_gpr.append(fevd_decomp[bcti_index][gpr_index])

        wti_to_bdi.append(fevd_decomp[wti_index][bdi_index])
        wti_to_bdti.append(fevd_decomp[wti_index][bdti_index])
        wti_to_bcti.append(fevd_decomp[wti_index][bcti_index])
        wti_to_wti.append(fevd_decomp[wti_index][wti_index])
        wti_to_gpr.append(fevd_decomp[wti_index][gpr_index])

        gpr_to_bdi.append(fevd_decomp[gpr_index][bdi_index])
        gpr_to_bdti.append(fevd_decomp[gpr_index][bdti_index])
        gpr_to_bcti.append(fevd_decomp[gpr_index][bcti_index])
        gpr_to_wti.append(fevd_decomp[gpr_index][wti_index])
        gpr_to_gpr.append(fevd_decomp[gpr_index][gpr_index])


        # 计算总溢出指数
        total_spillover.append((np.sum(fevd_decomp) - np.trace(fevd_decomp)) / 5)

        window_start_times.append(log_returns.index[i].to_timestamp())

    except AttributeError:
        print(f"处理窗口 {i} 时，fevd 对象没有 'decomp' 属性。")

# 计算平均方向性溢出指数
average_spillover_matrix = {
    'BDI_to_BDI': np.mean(bdi_to_bdi),
    'BDI_to_BDTI': np.mean(bdi_to_bdti),
    'BDI_to_BCTI': np.mean(bdi_to_bcti),
    'BDI_to_WTI': np.mean(bdi_to_wti),
    'BDI_to_GPR': np.mean(bdi_to_gpr),

    'BDTI_to_BDI': np.mean(bdti_to_bdi),
    'BDTI_to_BDTI': np.mean(bdti_to_bdti),
    'BDTI_to_BCTI': np.mean(bdti_to_bcti),
    'BDTI_to_WTI': np.mean(bdti_to_wti),
    'BDTI_to_GPR': np.mean(bdti_to_gpr),

    'BCTI_to_BDI': np.mean(bcti_to_bdi),
    'BCTI_to_BDTI': np.mean(bcti_to_bdti),
    'BCTI_to_BCTI': np.mean(bcti_to_bcti),
    'BCTI_to_WTI': np.mean(bcti_to_wti),
    'BCTI_to_GPR': np.mean(bcti_to_gpr),

    'WTI_to_BDI': np.mean(wti_to_bdi),
    'WTI_to_BDTI': np.mean(wti_to_bdti),
    'WTI_to_BCTI': np.mean(wti_to_bcti),
    'WTI_to_WTI': np.mean(wti_to_wti),
    'WTI_to_GPR': np.mean(wti_to_gpr),

    'GPR_to_BDI': np.mean(gpr_to_bdi),
    'GPR_to_BDTI': np.mean(gpr_to_bdti),
    'GPR_to_BCTI': np.mean(gpr_to_bcti),
    'GPR_to_WTI': np.mean(gpr_to_wti),
    'GPR_to_GPR': np.mean(gpr_to_gpr)
}

# 计算平均总溢出指数
average_total_spillover = np.mean(total_spillover)

# 输出平均方向性溢出指数
print("\n平均方向性溢出指数 (Average Spillover Index):")
for key, value in average_spillover_matrix.items():
    print(f"{key}: {value:.4f}")

# 输出平均总溢出指数
print("\n平均总溢出指数 (Average Total Spillover Index):")
print(f"Average Total Spillover Index: {average_total_spillover:.4f}")
# len all = 6239

# 输出每个滚动窗口自动拟合最佳的滞后阶数p
print(f"每个滚动窗口的p = {window_p}")

# ————————————————TO————————————————
AVG_BDI_To = []
AVG_BDTI_To = []
AVG_BCTI_To = []
AVG_WTI_To = []
AVG_GPR_To = []

for i in range(len(bdi_to_bdi)):
    sum_value = bdi_to_bdti[i] + bdi_to_bcti[i] + bdi_to_wti[i] + bdi_to_gpr[i]
    avg_value = sum_value / 4
    AVG_BDI_To.append(avg_value)

for i in range(len(bdi_to_bdi)):
    sum_value = bdti_to_bdi[i] + bdti_to_bcti[i] + bdti_to_wti[i] + bdti_to_gpr[i]
    avg_value = sum_value / 4
    AVG_BDTI_To.append(avg_value)

for i in range(len(bdi_to_bdi)):
    sum_value = bcti_to_bdi[i] + bcti_to_bdti[i] + bcti_to_wti[i] + bcti_to_gpr[i]
    avg_value = sum_value / 4
    AVG_BCTI_To.append(avg_value)

for i in range(len(bdi_to_bdi)):
    sum_value = wti_to_bdi[i] + wti_to_bdti[i] + wti_to_bcti[i] + wti_to_gpr[i]
    avg_value = sum_value / 4
    AVG_WTI_To.append(avg_value)

for i in range(len(bdi_to_bdi)):
    sum_value = gpr_to_bdi[i] + gpr_to_bdti[i] + gpr_to_bcti[i] + gpr_to_wti[i]
    avg_value = sum_value / 4
    AVG_GPR_To.append(avg_value)

# ————————————————TO————————————————



# ————————————————FROM————————————————
AVG_BDI_FROM = []
AVG_BDTI_FROM = []
AVG_BCTI_FROM = []
AVG_WTI_FROM = []
AVG_GPR_FROM = []

for i in range(len(bdi_to_bdi)):
    sum_value =  bdti_to_bdi[i] + bcti_to_bdi[i] + wti_to_bdi[i] + gpr_to_bdi[i]
    avg_value = sum_value / 4
    AVG_BDI_FROM.append(avg_value)

for i in range(len(bdi_to_bdi)):
    sum_value = bdi_to_bdti[i] + bcti_to_bdti[i] + wti_to_bdti[i] + gpr_to_bdti[i]
    avg_value = sum_value / 4
    AVG_BDTI_FROM.append(avg_value)

for i in range(len(bdi_to_bdi)):
    sum_value =  bdi_to_bcti[i] + bdti_to_bcti[i] +  wti_to_bcti[i] + gpr_to_bcti[i]
    avg_value = sum_value / 4
    AVG_BCTI_FROM.append(avg_value)

for i in range(len(bdi_to_bdi)):
    sum_value = bdi_to_wti[i] + bdti_to_wti[i] + bcti_to_wti[i] + gpr_to_wti[i]
    avg_value = sum_value / 4
    AVG_WTI_FROM.append(avg_value)


for i in range(len(bdi_to_bdi)):
    sum_value = bdi_to_gpr[i] + bdti_to_gpr[i] + bcti_to_gpr[i] + wti_to_gpr[i]
    avg_value = sum_value / 4
    AVG_GPR_FROM.append(avg_value)
# ————————————————FROM————————————————

plt.rcParams.update({
    # 'figure.dpi': 110,          # 核心：控制所有图像分辨率为300dpi
    # 'savefig.dpi': 300,         # 保存图像时同样使用300dpi（避免保存后分辨率降低）
    'font.size': 20,            # 全局字体大小（避免标题/标签字体不协调）
    'font.sans-serif': ['Arial'],  # 统一字体（避免中文乱码，若需中文可换'WenQuanYi Zen Hei'）
    # 'axes.grid': False,          # 添加网格线（便于读取数值，可根据需求关闭）
    # 'grid.alpha': 0.3,          # 网格线透明度（避免遮挡曲线）
    # 'axes.spines.top': False,   # 隐藏上边框（简洁风格）
    # 'axes.spines.right': False, # 隐藏右边框
    # 'xtick.rotation': 45,       # 统一x轴日期旋转45度（无需重复设置）
    # 'xtick.direction': 'in',    # x轴刻度向内（美观）
    # 'ytick.direction': 'in'     # y轴刻度向内
})




# 总溢出指数图
plt.figure(figsize=(12, 8))  # 显示大小
# plt.subplot(2, 2, 1)
plt.plot(window_start_times, total_spillover, label=' total_spillover ', color='blue')
plt.title(' total_spillover')
plt.xlabel('Date')
plt.ylabel('Mean Spillover Index')
plt.xticks(rotation=45)
plt.legend()
plt.gcf().subplots_adjust(bottom=0.15)


# # 绘制 BDI-TO 图
# ****************************多个图*******************************

# 设置全局样式
plt.rcParams.update({
    'font.size': 20,  # 全局字体大小
    'font.sans-serif': ['Arial'],  # 统一字体
})

# 定义颜色方案，与之前风格保持一致
colors = [
    ('#CC3333', '#990000'),  # 红色系
    ('#1A75FF', '#0047B3'),  # 蓝色系
    ('#33AA33', '#007700'),  # 绿色系
    ('#FF9900', '#CC7700'),  # 橙色系
    ('#CC3399', '#990066')  # 粉色系
]

# 准备数据和标签
data = [
    (AVG_BDI_To, 'BDI to', 'BDI_To'),
    (AVG_BDTI_To, 'BDTI to', 'BDTI_To'),
    (AVG_BCTI_To, 'BCTI to', 'BCTI_To'),
    (AVG_WTI_To, 'WTI to', 'WTI_To'), # WTI 与 BRENT 替换
    (AVG_GPR_To, 'GPR to', 'GPR_To')
]

# 创建两行三列布局的图表
plt.figure(figsize=(24, 14))  # 调整整体大小
plt.subplots_adjust(
    top=0.92,  # 顶部边距，为标题块留出空间
    hspace=0.4,  # 垂直方向子图间距
    wspace=0.3,  # 水平方向子图间距
    left=0.08,  # 左侧边距
    right=0.95,  # 右侧边距
    bottom=0.15  # 底部边距
)

for i, (data_series, label, title) in enumerate(data):
    # 创建子图（2行3列）
    ax = plt.subplot(2, 3, i + 1)

    # 获取对应颜色（使用颜色组的第一个颜色）
    fill_color = colors[i % len(colors)][0]

    # 绘制填充曲线
    plt.fill_between(
        window_start_times,
        data_series,
        label=label,
        color=fill_color,
        alpha=0.3
    )

    # 同时绘制线条使曲线更清晰
    plt.plot(
        window_start_times,
        data_series,
        color=colors[i % len(colors)][1],  # 使用深色线条
        linewidth=1.5
    )

    # 子图标签与格式
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mean Spillover Index', fontsize=18)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc='upper right')

    # 清除默认标题，使用自定义灰色标题块
    plt.title('')

    # 获取子图位置，创建灰色标题块
    bbox = ax.get_position()
    title_ax = plt.axes([
        bbox.x0,
        bbox.y0 + bbox.height,  # 位于子图上方
        bbox.width,
        0.04  # 标题块高度
    ])

    # 添加带边框的灰色背景
    title_ax.add_patch(plt.Rectangle(
        (0, 0), 1, 1,
        facecolor='lightgray',
        edgecolor='black',
        linewidth=1.2,
        alpha=0.7
    ))

    # 添加标题文本
    title_ax.text(
        0.5, 0.5,
        title,
        ha='center',
        va='center',
        fontsize=16,
        transform=title_ax.transAxes
    )
    title_ax.axis('off')  # 隐藏标题轴

# 第6个子图位置留空（因为只有5个数据系列）
if len(data) < 6:
    ax_empty = plt.subplot(2, 3, 6)
    ax_empty.axis('off')  # 隐藏空轴

plt.show()

# ****************************多个图*******************************


# ****************************单个图*******************************
# plt.figure(figsize=(12, 8))  # 显示大小
# plt.fill_between(window_start_times, AVG_BDI_To, label='BDI to', color='blue', alpha=0.3)
# plt.title('BDI_To')
# plt.xlabel('Date')
# plt.ylabel('Mean Spillover Index')
# plt.xticks(rotation=45)
# plt.legend()
# plt.gcf().subplots_adjust(bottom=0.15)
#
# # 绘制 BDTI-TO 图
# plt.figure(figsize=(12, 8))  # 显示大小
# plt.fill_between(window_start_times, AVG_BDTI_To, label='BDTI to', color='blue', alpha=0.3)
# plt.title('BDTI_To')
# plt.xlabel('Date')
# plt.ylabel('Mean Spillover Index')
# plt.xticks(rotation=45)
# plt.legend()
# plt.gcf().subplots_adjust(bottom=0.15)
#
# # 绘制 BCTI-TO 图
# plt.figure(figsize=(12, 8))  # 显示大小
# plt.fill_between(window_start_times, AVG_BCTI_To, label='BCTI to', color='blue', alpha=0.3)
# plt.title('BCTI_To')
# plt.xlabel('Date')
# plt.ylabel('Mean Spillover Index')
# plt.xticks(rotation=45)
# plt.legend()
# plt.gcf().subplots_adjust(bottom=0.15)
#
# # 绘制 WTI-TO 图
# plt.figure(figsize=(12, 8))  # 显示大小
# plt.fill_between(window_start_times, AVG_WTI_To, label='WTI to', color='blue', alpha=0.3)
# plt.title('WTI_To')
# plt.xlabel('Date')
# plt.ylabel('Mean Spillover Index')
# plt.xticks(rotation=45)
# plt.legend()
# plt.gcf().subplots_adjust(bottom=0.15)
#
# # 绘制 GPR-TO 图
# plt.figure(figsize=(12, 8))  # 显示大小
# plt.fill_between(window_start_times, AVG_GPR_To, label='GPR_To', color='blue', alpha=0.3)
# plt.title('GPR_To')
# plt.xlabel('Date')
# plt.ylabel('Mean Spillover Index')
# plt.xticks(rotation=45)
# plt.legend()
# plt.gcf().subplots_adjust(bottom=0.15)
# ****************************单个图*******************************

# ——————————————————split————————————————————

# 绘制 BDI-FROM 图

# ****************************多个图*******************************
# 设置全局样式
plt.rcParams.update({
    'font.size': 20,  # 全局字体大小
    'font.sans-serif': ['Arial'],  # 统一字体
})

# 定义颜色方案，与之前风格保持一致
colors = [
    ('#CC3333', '#990000'),  # 红色系
    ('#1A75FF', '#0047B3'),  # 蓝色系
    ('#33AA33', '#007700'),  # 绿色系
    ('#FF9900', '#CC7700'),  # 橙色系
    ('#CC3399', '#990066')  # 粉色系
]

# 准备FROM类型的数据和标签
data = [
    (AVG_BDI_FROM, 'BDI from', 'BDI_FROM'),
    (AVG_BDTI_FROM, 'BDTI from', 'BDTI_FROM'),
    (AVG_BCTI_FROM, 'BCTI from', 'BCTI_FROM'),
    (AVG_WTI_FROM, 'WTI from', 'WTI_FROM'), #  WTI 与 BRENT 替换
    (AVG_GPR_FROM, 'GPR from', 'GPR_FROM')
]

# 创建两行三列布局的图表
plt.figure(figsize=(24, 14))  # 调整整体大小
plt.subplots_adjust(
    top=0.92,  # 顶部边距，为标题块留出空间
    hspace=0.4,  # 垂直方向子图间距
    wspace=0.3,  # 水平方向子图间距
    left=0.08,  # 左侧边距
    right=0.95,  # 右侧边距
    bottom=0.15  # 底部边距
)

for i, (data_series, label, title) in enumerate(data):
    # 创建子图（2行3列）
    ax = plt.subplot(2, 3, i + 1)

    # 获取对应颜色（使用颜色组的第一个颜色）
    fill_color = colors[i % len(colors)][0]

    # 绘制填充曲线
    plt.fill_between(
        window_start_times,
        data_series,
        label=label,
        color=fill_color,
        alpha=0.3
    )

    # 同时绘制线条使曲线更清晰
    plt.plot(
        window_start_times,
        data_series,
        color=colors[i % len(colors)][1],  # 使用深色线条
        linewidth=1.5
    )

    # 子图标签与格式
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mean Spillover Index', fontsize=18)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc='upper right')

    # 清除默认标题，使用自定义灰色标题块
    plt.title('')

    # 获取子图位置，创建灰色标题块
    bbox = ax.get_position()
    title_ax = plt.axes([
        bbox.x0,
        bbox.y0 + bbox.height,  # 位于子图上方
        bbox.width,
        0.04  # 标题块高度
    ])

    # 添加带边框的灰色背景
    title_ax.add_patch(plt.Rectangle(
        (0, 0), 1, 1,
        facecolor='lightgray',
        edgecolor='black',
        linewidth=1.2,
        alpha=0.7
    ))

    # 添加标题文本
    title_ax.text(
        0.5, 0.5,
        title,
        ha='center',
        va='center',
        fontsize=16,
        transform=title_ax.transAxes
    )
    title_ax.axis('off')  # 隐藏标题轴

# 第6个子图位置留空（因为只有5个数据系列）
if len(data) < 6:
    ax_empty = plt.subplot(2, 3, 6)
    ax_empty.axis('off')  # 隐藏空轴

plt.show()

# ****************************多个图*******************************


# # ****************************单个图*******************************
# plt.figure(figsize=(12, 8))  # 显示大小
# plt.fill_between(window_start_times, AVG_BDI_FROM, label='BDI from', color='blue', alpha=0.3)
# plt.title('BDI_FROM')
# plt.xlabel('Date')
# plt.ylabel('Mean Spillover Index')
# plt.xticks(rotation=45)
# plt.legend()
# plt.gcf().subplots_adjust(bottom=0.15)
#
# # 绘制 BDTI-FROM 图
# plt.figure(figsize=(12, 8))  # 显示大小
# plt.fill_between(window_start_times, AVG_BDTI_FROM, label='BDTI from', color='blue', alpha=0.3)
# plt.title('BDTI_FROM')
# plt.xlabel('Date')
# plt.ylabel('Mean Spillover Index')
# plt.xticks(rotation=45)
# plt.legend()
# plt.gcf().subplots_adjust(bottom=0.15)
#
# # 绘制 BCTI-FROM 图
# plt.figure(figsize=(12, 8))  # 显示大小
# plt.fill_between(window_start_times, AVG_BCTI_FROM, label='BCTI from', color='blue', alpha=0.3)
# plt.title('BCTI_FROM')
# plt.xlabel('Date')
# plt.ylabel('Mean Spillover Index')
# plt.xticks(rotation=45)
# plt.legend()
# plt.gcf().subplots_adjust(bottom=0.15)
#
# # 绘制 WTI-FROM 图
# plt.figure(figsize=(12, 8))  # 显示大小
# plt.fill_between(window_start_times, AVG_WTI_FROM, label='WTI from', color='blue', alpha=0.3)
# plt.title('WTI_FROM')
# plt.xlabel('Date')
# plt.ylabel('Mean Spillover Index')
# plt.xticks(rotation=45)
# plt.legend()
# plt.gcf().subplots_adjust(bottom=0.15)
#
# # 绘制 WTI-FROM 图
# plt.figure(figsize=(12, 8))  # 显示大小
# plt.fill_between(window_start_times, AVG_GPR_FROM, label='GPR_FROM', color='blue', alpha=0.3)
# plt.title('GPR_FROM')
# plt.xlabel('Date')
# plt.ylabel('Mean Spillover Index')
# plt.xticks(rotation=45)
# plt.legend()
#
# plt.gcf().subplots_adjust(bottom=0.15)
# # plt.tight_layout()  # 自动调整布局
# plt.show()
# # ****************************单个图*******************************