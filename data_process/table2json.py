import pandas as pd
import json


def excel2json(origin_file, target_file, origin_sheet_name=0):
    """
    save excel文件为json
    Args:
    origin_file: 原始excel文件
    target_file: 目标json文件
    """

    # 读取 Excel，同时自动解析日期类型
    df = pd.read_excel(origin_file, sheet_name=origin_sheet_name, parse_dates=True)

    # 确保日期类型被转换为 ISO 格式字符串（如：2024-05-21T15:30:00）
    def convert_value(val):
        if pd.isna(val):
            return None
        elif isinstance(val, pd.Timestamp):
            return val.isoformat()
        elif isinstance(val, pd.Timedelta):
            return str(val)
        elif isinstance(val, (int, float, bool, str)):
            return val
        else:
            return str(val)

    # 应用转换
    json_ready = df.map(convert_value).to_dict(orient="records")

    # 保存为 JSON 文件
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(json_ready, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # QA data
    excel2json(origin_file='../resources/datas/QA_docs/QA_with_legal_basis.xlsx',
               target_file='../resources/datas/QA_docs/QA_with_legal_basis.json')

    # 标的物数据
    excel2json(origin_file='../resources/datas/have_data/招标采购标的物信息提取训练数据.xlsx',
               target_file='../knowledge/raw_knowledge/bid_cases/biaodewu_info.json',
               origin_sheet_name=1)
