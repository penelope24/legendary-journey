"""
ZigZag分析核心模块

提供简化版的ZigZag算法实现，用于识别价格序列中的主要转折点。
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from enum import Enum, auto
from datetime import datetime
import matplotlib.pyplot as plt
from src.data import StockInfoFetcher, Period

# 类型别名
DatePrice = Tuple[str, float]  # (日期字符串, 价格)元组
DateType = Union[str, datetime, pd.Timestamp, np.datetime64]  # 可能的日期类型

# 定义趋势枚举类
class Trend(Enum):
    """价格趋势的枚举表示"""
    UNDEFINED = auto()  # 趋势未确定
    UP = auto()         # 上升趋势
    DOWN = auto()       # 下降趋势
    
    def __str__(self) -> str:
        return self.name


class ZigZag:
    """ZigZag 分析类，简化版"""
    
    # 默认设置
    DEFAULT_THRESHOLD = 0.05  # 默认波动阈值，5%
    DATE_FORMAT = "%Y%m%d"    # 日期格式
        
    def __init__(self, df: pd.DataFrame, price_col: str = 'close'):
        """
        初始化ZigZag分析器
        
        Args:
            df: 价格数据DataFrame，必须包含'trade_date'列和price_col指定的价格列
            price_col: 价格列名，默认为'close'
        """
        self.price_col = price_col
        self.df = self._preprocess_dataframe(df)
    
    def _format_date(self, date: DateType) -> str:
        """
        格式化日期为统一格式
        
        Args:
            date: 日期对象，可以是字符串、datetime、Timestamp或numpy.datetime64
            
        Returns:
            str: 格式化后的日期字符串（YYYYMMDD）
        """
        if isinstance(date, str):
            # 尝试解析字符串日期
            try:
                # 如果已经是正确格式，直接返回
                if len(date) == 8 and date.isdigit():
                    return date
                # 尝试解析为日期格式
                parsed_date = pd.to_datetime(date)
                return parsed_date.strftime(self.DATE_FORMAT)
            except:
                raise ValueError(f"无法解析日期字符串: {date}")
        elif isinstance(date, (datetime, pd.Timestamp)):
            return date.strftime(self.DATE_FORMAT)
        elif isinstance(date, np.datetime64):
            return pd.Timestamp(date).strftime(self.DATE_FORMAT)
        else:
            raise ValueError(f"不支持的日期格式: {type(date)}")
    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理数据, 按时间升序排序
        
        Args:
            df: 需要预处理的DataFrame
            
        Returns:
            pd.DataFrame: 预处理后的DataFrame
        """
        df = df.copy()
        # 确保必要的列存在
        if 'trade_date' not in df.columns:
            raise KeyError("DataFrame必须包含'trade_date'列")
        if self.price_col not in df.columns:
            raise KeyError(f"DataFrame必须包含'{self.price_col}'列")
        
        # 确保价格列是数值类型
        df[self.price_col] = pd.to_numeric(df[self.price_col])
        
        # 确保trade_date列是日期类型
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 按日期排序，升序
        return df.sort_values('trade_date').reset_index(drop=True)
        
    def analyze(self, threshold: float = DEFAULT_THRESHOLD, vectorize: bool = True) -> Dict[str, Any]:
        """
        计算价格序列的ZigZag折线点位
        
        Args:
            threshold: 转折点判定的阈值（以小数表示的百分比，如0.05表示5%）
            vectorize: 是否使用向量化方法（更高效）
        
        Returns:
            Dict[str, Any]: 包含ZigZag分析结果的字典
        """
        # 参数验证
        if not 0 < threshold < 1:
            raise ValueError(f"阈值必须在0和1之间，当前值: {threshold}")
        
        # 根据参数选择计算方法
        if vectorize:
            zigzag_points = self._analyze_vectorized(threshold)
        else:
            zigzag_points = self._analyze_simple(threshold)
        
        # 获取高点和低点
        high_points, low_points = self._get_top_k_points(zigzag_points)
        
        # 返回结果
        return {
            'points': zigzag_points,
            'high_points': high_points,
            'low_points': low_points,
            'threshold': threshold,
            'method': 'vectorized' if vectorize else 'simple'
        }
    
    def _analyze_simple(self, threshold: float) -> List[DatePrice]:
        """
        简单实现的ZigZag计算（循环方式）
        
        Args:
            threshold: 转折点判定的阈值
            
        Returns:
            List[DatePrice]: ZigZag点位列表
        """
        df = self.df
        price_series = df[self.price_col].values
        dates = df['trade_date'].apply(self._format_date).values
        n = len(price_series)
        
        if n < 2:
            return [(dates[0], price_series[0])] if n == 1 else []
       
        # 初始化结果和状态
        result = [(dates[0], price_series[0])]
        trend = Trend.UNDEFINED  # 初始状态为未确定
        
        for i in range(1, n):
            last_price = result[-1][1]
            curr_price = price_series[i]
            
            # 计算涨跌幅
            change_pct = (curr_price - last_price) / last_price
            
            # 判断趋势
            if trend == Trend.UNDEFINED:
                # 初始趋势确认
                if abs(change_pct) >= threshold:
                    trend = Trend.UP if change_pct > 0 else Trend.DOWN
                    result.append((dates[i], curr_price))
            elif trend == Trend.UP:  # 上升趋势
                if curr_price > last_price:
                    # 继续上升，更新高点
                    result[-1] = (dates[i], curr_price)
                elif abs(change_pct) >= threshold:
                    # 反转为下降趋势
                    trend = Trend.DOWN
                    result.append((dates[i], curr_price))
            else:  # 下降趋势
                if curr_price < last_price:
                    # 继续下降，更新低点
                    result[-1] = (dates[i], curr_price)
                elif abs(change_pct) >= threshold:
                    # 反转为上升趋势
                    trend = Trend.UP
                    result.append((dates[i], curr_price))
        
        # 添加最后一个点（如果趋势已确定）
        if trend != Trend.UNDEFINED and result[-1][0] != dates[-1]:
            result.append((dates[-1], price_series[-1]))
        
        return result
    
    def _analyze_vectorized(self, threshold: float) -> List[DatePrice]:
        """
        向量化实现的ZigZag计算（更高效）
        
        Args:
            threshold: 转折点判定的阈值
            
        Returns:
            List[DatePrice]: ZigZag点位列表
        """
        df = self.df
        price_series = df[self.price_col].values
        dates = df['trade_date'].apply(self._format_date).values
        n = len(price_series)
        
        if n < 2:
            return [(dates[0], price_series[0])] if n == 1 else []
        
        # 创建布尔掩码数组，标记要保留的点
        extrema = np.zeros(n, dtype=bool)
        extrema[0] = True  # 首点总是包含的

        # 初始状态
        trend = Trend.UNDEFINED
        last_extreme_idx = 0
        
        # 预先计算所有点相对于第一个点的变化百分比
        changes_from_first = (price_series - price_series[0]) / price_series[0]
        
        # 确定初始趋势
        for i in range(1, n):
            if abs(changes_from_first[i]) >= threshold:
                trend = Trend.UP if changes_from_first[i] > 0 else Trend.DOWN
                extrema[i] = True
                last_extreme_idx = i
                break
        
        # 如果没有找到初始趋势，直接返回首点
        if trend == Trend.UNDEFINED:
            return [(dates[0], price_series[0])]
        
        # 处理后续点
        i = last_extreme_idx + 1
        while i < n:
            # 计算相对于上一个极点的变化
            curr_price = price_series[i]
            last_price = price_series[last_extreme_idx]
            change_pct = (curr_price - last_price) / last_price
            
            if trend == Trend.UP:  # 当前是上升趋势
                if curr_price > last_price:
                    # 新高点，更新现有高点
                    extrema[last_extreme_idx] = False
                    extrema[i] = True
                    last_extreme_idx = i
                elif change_pct <= -threshold:
                    # 下跌超过阈值，趋势反转
                    trend = Trend.DOWN
                    extrema[i] = True
                    last_extreme_idx = i
            else:  # 当前是下降趋势
                if curr_price < last_price:
                    # 新低点，更新现有低点
                    extrema[last_extreme_idx] = False
                    extrema[i] = True
                    last_extreme_idx = i
                elif change_pct >= threshold:
                    # 上涨超过阈值，趋势反转
                    trend = Trend.UP
                    extrema[i] = True
                    last_extreme_idx = i
            i += 1
            
        # 提取结果
        result_indices = np.where(extrema)[0]
        result = [(dates[i], price_series[i]) for i in result_indices]
        
        # 添加最后一个点（如果不在结果中）
        if result[-1][0] != dates[-1]:
            result.append((dates[-1], price_series[-1]))
        
        return result
    
    def _get_top_k_points(self, zigzag_points: List[DatePrice], k: int = 5) -> Tuple[List[DatePrice], List[DatePrice]]:
        """
        获取ZigZag点位列表中前k个高点和前k个低点
        
        Args:
            zigzag_points: ZigZag分析结果点位列表
            k: 需要返回的每种点位数量
            
        Returns:
            Tuple[List[DatePrice], List[DatePrice]]: (前k个最高点列表, 前k个最低点列表)
        """
        if not zigzag_points or len(zigzag_points) < 2:
            return [], []
        
        # 分离高点和低点
        high_points = []
        low_points = []
        
        for i in range(len(zigzag_points)):
            is_high = False
            is_low = False
            
            # 第一个点：如果比第二个点高，就是高点；否则是低点
            if i == 0:
                if zigzag_points[0][1] > zigzag_points[1][1]:
                    is_high = True
                else:
                    is_low = True
            # 最后一个点：如果比倒数第二个点高，就是高点；否则是低点
            elif i == len(zigzag_points) - 1:
                if zigzag_points[i][1] > zigzag_points[i-1][1]:
                    is_high = True
                else:
                    is_low = True
            # 中间点：根据前后点位判断
            else:
                # 如果当前点比前后点都高，就是高点
                if zigzag_points[i][1] > zigzag_points[i-1][1] and zigzag_points[i][1] > zigzag_points[i+1][1]:
                    is_high = True
                # 如果当前点比前后点都低，就是低点
                elif zigzag_points[i][1] < zigzag_points[i-1][1] and zigzag_points[i][1] < zigzag_points[i+1][1]:
                    is_low = True
            
            if is_high:
                high_points.append(zigzag_points[i])
            elif is_low:
                low_points.append(zigzag_points[i])
        
        # 按价格排序（高点降序，低点升序）
        high_points.sort(key=lambda x: x[1], reverse=True)  # 高点降序
        low_points.sort(key=lambda x: x[1])  # 低点升序
        
        # 返回前k个高点和低点
        return high_points[:k], low_points[:k]
    
def find_highest_price_point(df: pd.DataFrame, price_col: str = 'close') -> Tuple[str, float]:
    """
    从DataFrame中找出价格最高的点
    
    Args:
        df: 包含价格数据的DataFrame
        price_col: 价格列名，默认为'close'
        
    Returns:
        Tuple[str, float]: (日期字符串, 最高价格)
    """
    # 获取最高价格的索引
    max_idx = df[price_col].idxmax()
    
    # 获取对应的日期和价格
    max_date = df.loc[max_idx, 'trade_date']
    max_price = df.loc[max_idx, price_col]
    
    # 将日期格式化为字符串
    if isinstance(max_date, (pd.Timestamp, datetime, np.datetime64)):
        max_date_str = pd.Timestamp(max_date).strftime('%Y%m%d')
    else:
        max_date_str = str(max_date)
    
    return max_date_str, float(max_price)
    
if __name__ == "__main__":
    # 获取茅台从20100101至今的数据并演示zigzag分析的结果
    # 创建茅台股票数据获取器 
    stock_code = '000166.SZ'  # 茅台股票代码
    fetcher = StockInfoFetcher(stock_code)
    
    # 获取茅台股票数据（从2010年到现在）
    start_date = '20140101'
    # end_date = datetime.now().strftime('%Y%m%d') 
    end_date = '20170101'
    df = fetcher.get_kline_data(Period.WEEKLY, start_date, end_date)
    
    # 初始化ZigZag分析器
    zigzag = ZigZag(df)
    
    # 执行分析（使用6%阈值）
    result = zigzag.analyze(threshold=0.06)
    
    # 提取分析结果
    points = result['points']
    high_points = result['high_points']
    low_points = result['low_points']
    
    # 打印结果
    print(f"找到 {len(points)} 个ZigZag转折点")
    print(f"前5个高点:")
    for date, price in high_points[:5]:
        print(f"  {date}: {price:.2f}")
    print(f"前5个低点:")
    for date, price in low_points[:5]:
        print(f"  {date}: {price:.2f}")
    
    # 可视化
    plt.figure(figsize=(12, 6))
    
    # 绘制原始价格
    dates = pd.to_datetime(df['trade_date'])
    plt.plot(dates, df['close'], 'gray', alpha=0.5, label='Moutai Weekly Close')
    
    # 绘制ZigZag折线
    zigzag_dates = [pd.to_datetime(date) for date, _ in points]
    zigzag_prices = [price for _, price in points]
    plt.plot(zigzag_dates, zigzag_prices, 'r-', linewidth=2, label='ZigZag Line')
    
    # 绘制高点和低点
    high_dates = [pd.to_datetime(date) for date, _ in high_points[:5]]
    high_prices = [price for _, price in high_points[:5]]
    plt.scatter(high_dates, high_prices, c='red', s=100, marker='^', label='Major Highs')
    
    low_dates = [pd.to_datetime(date) for date, _ in low_points[:5]]
    low_prices = [price for _, price in low_points[:5]]
    plt.scatter(low_dates, low_prices, c='green', s=100, marker='v', label='Major Lows')
    
    # 设置图表
    plt.title(f'Moutai Stock ZigZag Analysis (2010-2023, Threshold: 6%)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 显示图表
    plt.tight_layout()
    plt.show()
    