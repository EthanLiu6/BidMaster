import logging
import sys

from scripts import config


def set_log():
    # 基础配置（写入文件）
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=config.log_file,  # 日志文件路径
        filemode='a'
    )
    # 禁用 pdfminer 的特定警告
    logging.getLogger('pdfminer').setLevel(logging.ERROR)

    # 添加一个额外的控制台处理器（输出到终端）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(console_handler)
