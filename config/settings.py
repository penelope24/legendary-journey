"""全局配置参数"""

# 股票代码
STOCK_CODE = "600519.SH"

# 股票名称
STOCK_NAME = "贵州茅台" 

# 股票数据源
STOCK_DATA_SOURCE = "tushare"

# 股票数据源的API key
STOCK_DATA_SOURCE_API_KEY = "8465008d5e8f1b8e589818ffeb079887b6decd4d11b540d65192fd1f"

# 缓存设置
CACHE_DIR = ".cache"  # 缓存根目录，相对于项目根目录
CACHE_EXPIRE_DAYS = 1  # 缓存默认过期天数