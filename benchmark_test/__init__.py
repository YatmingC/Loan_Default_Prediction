"""benchmark_test模块初始化文件

延迟导入：避免 import benchmark_test 时强制拉入 json_repair 等重量级依赖，
使 test_data_loading.py 等轻量脚本可以单独使用 data_loader 子模块。
"""


def __getattr__(name):
    if name == "SBADataLoader":
        from .data_loader_sba import SBADataLoader
        return SBADataLoader
    if name == "LendingClubDataLoader":
        from .data_loader_lendingclub import LendingClubDataLoader
        return LendingClubDataLoader
    if name == "HomeCreditDataLoader":
        from .data_loader_homecredit import HomeCreditDataLoader
        return HomeCreditDataLoader
    if name == "BaselineInferenceEngine":
        from .baseline_inference import BaselineInferenceEngine
        return BaselineInferenceEngine
    raise AttributeError(f"module 'benchmark_test' has no attribute {name!r}")


__all__ = [
    'SBADataLoader',
    'LendingClubDataLoader',
    'HomeCreditDataLoader',
    'BaselineInferenceEngine',
]
