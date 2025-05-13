from pathlib import Path


def find_project_root():
    # 当前脚本路径
    current_file = Path(__file__).resolve()

    # 项目根目录（BidMaster 是根）
    project_root = current_file.parents[1]  # 逐级向上：utils -> BidMaster
    return project_root
