"""
股票数据获取模块

提供股票数据的获取和基本处理功能，包括：
- 周K线数据获取
- 基本数据处理
"""

import os
from typing import List
from enum import Enum

import pandas as pd
import tushare as ts
from dotenv import load_dotenv


class Period(Enum):
    """K线周期"""
    DAILY = 'daily'     # 日线
    WEEKLY = 'weekly'   # 周线
    MONTHLY = 'monthly' # 月线
    
class Adj(Enum):
    """复权类型"""
    QFQ = 'qfq' # 前复权
    HFQ = 'hfq' # 后复权
    NONE = 'none' # 不复权

class StockInfoFetcher:
    """
    股票数据获取器
    
    用于获取和处理单只股票的各类数据，包括K线数据、基本面数据等。
    
    Attributes:
        code (str): 股票代码
        pro (ts.pro_api): Tushare Pro API接口
    """
    
    def __init__(self, code: str):
        """
        初始化股票数据获取器
        
        Args:
            code (str): 股票代码，如 '600519.SH'
        """
        self.code = code
        self._init_ts()
    
    def _init_ts(self) -> None:
        """初始化Tushare接口"""
        load_dotenv()
        ts.set_token(os.getenv('TUSHARE_TOKEN'))
        self.pro = ts.pro_api()
    
    def get_kline_data(
        self,
        period: Period,
        start_date: str,
        end_date: str,
        fields: List[str] = [
            'trade_date',
            'open',
            'high',
            'low',
            'close',
            'vol'
        ]
    ) -> pd.DataFrame:
        """
        获取指定周期的K线数据
        
        Args:
            period (Period): K线周期，可选值：DAILY（日线）, WEEKLY（周线）, MONTHLY（月线）
            start_date (str): 起始日期，格式：YYYYMMDD
            end_date (str): 结束日期，格式：YYYYMMDD
            fields (List[str], optional): 需要获取的字段列表。默认包含基本的OHLCV数据。
        
        Returns:
            pd.DataFrame: 包含指定字段的K线数据
            
        Examples:
            >>> fetcher = StockInfoFetcher('600519.SH')
            >>> # 获取周线数据
            >>> weekly_data = fetcher.get_kline_data(Period.WEEKLY, '20210101', '20210630')
            >>> # 获取日线数据
            >>> daily_data = fetcher.get_kline_data(Period.DAILY, '20210101', '20210630')
        """
        # 获取对应的API方法
        api_method = getattr(self.pro, period.value)
        
        # 调用API获取数据
        df = api_method(
            ts_code=self.code,
            start_date=start_date,
            end_date=end_date,
            fields=fields
        )
        return df
    
    def get_kline_data_adj(self, period: Period, adj: Adj, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指定周期的K线数据（复权）
        
        Args:
            period (Period): K线周期，可选值：DAILY（日线）, WEEKLY（周线）, MONTHLY（月线）
            start_date (str): 起始日期，格式：YYYYMMDD
            end_date (str): 结束日期，格式：YYYYMMDD
            fields (List[str], optional): 需要获取的字段列表。默认包含基本的OHLCV数据。
        
        Returns:
            pd.DataFrame: 包含指定字段的K线数据
        """
        period_map = {
            Period.DAILY: 'D',
            Period.WEEKLY: 'W',
            Period.MONTHLY: 'M'
        }
        freq = period_map[period]
        df = ts.pro_bar(
            ts_code=self.code,
            adj=adj.value,
            start_date=start_date,
            end_date=end_date,
            freq=freq
        )
        return df
    
    def get_ma_lines(self, start_date: str, end_date: str, ma_periods: List[int] = [5, 22]) -> pd.DataFrame:
        """
        获取指定周期的MA线数据
        
        Args:
            start_date (str): 起始日期，格式：YYYYMMDD
            end_date (str): 结束日期，格式：YYYYMMDD
            ma_periods (List[int], optional): MA周期列表。默认包含5日均线和22日均线。
        
        Returns:
            pd.DataFrame: 包含指定周期的MA线数据
        """
        df = ts.pro_bar(
            ts_code=self.code,
            start_date=start_date,
            end_date=end_date,
            ma=ma_periods
        )
        return df
    
    def get_factors(self, start_date: str, end_date: str, factors: List[str] = ['tor', 'vr']) -> pd.DataFrame:
        """
        获取指定周期的因子数据
        
        Args:
            start_date (str): 起始日期，格式：YYYYMMDD
            end_date (str): 结束日期，格式：YYYYMMDD
            factors (List[str], optional): 需要获取的因子列表。默认包含tor和vr。
        
        Returns:
            pd.DataFrame: 包含指定周期的因子数据
        """
        df = ts.pro_bar(
            ts_code=self.code,
            start_date=start_date,
            end_date=end_date,
            factors=factors
        )
        return df
    
if __name__ == '__main__':
    ts_code = '600519.SH'
    start_date = '20210101'
    end_date = '20250311'
    
    # 创建数据获取器实例
    stock_info_fetcher = StockInfoFetcher(ts_code)
    
    # 获取不同周期的数据
    weekly_data = stock_info_fetcher.get_kline_data(Period.WEEKLY, start_date, end_date)
    print("\n周线数据:")
    print(weekly_data)
    
    daily_data = stock_info_fetcher.get_kline_data(Period.DAILY, start_date, end_date)
    print("\n日线数据:")
    print(daily_data)
    
    monthly_data = stock_info_fetcher.get_kline_data(Period.MONTHLY, start_date, end_date)
    print("\n月线数据:")
    print(monthly_data)