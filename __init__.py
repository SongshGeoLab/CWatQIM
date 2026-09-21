#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""Water Quota Incentive Model (CWatQIM) Package.

A agent-based model for simulating Yellow River water quota allocation and
analyzing policy incentives. This package implements an agent-based model (ABM)
that simulates the interactions between provinces, cities, and farmers in the
Yellow River Basin, focusing on water quota compliance and social learning
mechanisms.

The model is built on the ABSESpy framework and integrates with AquaCrop for
crop yield simulation. It can be published independently as it contains only
the core model components without analysis dependencies.

Main Components:
    - CWatQIModel: The main model class that orchestrates the simulation
    - City: City-level agents representing irrigation units
    - Province: Province-level agents managing water quota allocation
    - Core utilities: Algorithms, data loaders, and payoff calculations

Example:
    Basic usage of the model:

    ```python
    from cwatqim import CWatQIModel
    from hydra import compose, initialize

    with initialize(config_path="config", version_base=None):
        cfg = compose(config_name="demo")
        model = CWatQIModel(parameters=cfg)
        model.setup()
        for _ in range(10):
            model.step()
        model.end()
    ```

Note:
    This package is designed to be independent of analysis tools. For result
    analysis, use the separate `water_quota_analysis` package.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - 只为类型检查器与 IDE 提供符号
    # noqa 是必需的：真正的导出走下面的 `__getattr__`，而 `__all__` 由
    # `_LAZY` 派生（不再是字面量列表），flake8 认不出这几个是再导出。
    from .agents import City, Farmer, Province  # noqa: F401
    from .model import CWatQIModel  # noqa: F401

# 这四个符号必须**惰性**导出，不能在模块顶层 import（见 issue #48）。
#
# `aquacrop/__init__.py` 用 `if not '-m' in sys.argv:` 决定要不要导出自己的
# 全部符号。而 CPython 在 `python -m pkg` 时，执行包 `__init__.py` 期间
# `sys.argv[0]` 就是字面量 `'-m'`——runpy 要等模块解析完才把它换成文件路径。
# 于是顶层 import 会在那一刻拉起 aquacrop，撞上它的空导出分支，
# `python -m cwatqim` 直接崩在 `ImportError: cannot import name 'Crop'`，
# 而 `import cwatqim` 一切正常。
#
# 惰性化之后，真正的导入推迟到 `__main__` 里——那时 runpy 已经把 argv[0]
# 换成文件路径，aquacrop 正常导出。附带好处是 `import cwatqim` 不再无条件
# 付出 aquacrop 的导入代价。
_LAZY = {
    "City": ".agents",
    "Farmer": ".agents",
    "Province": ".agents",
    "CWatQIModel": ".model",
}

# 对外承诺与惰性映射共用一个真源，不必手动保持两份同步
__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    """Import the public model classes on first use.

    Args:
        name: Attribute requested from this package.

    Returns:
        The requested class.

    Raises:
        AttributeError: For any name this package does not export.
    """
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(_LAZY[name], __name__)
    value = getattr(module, name)
    globals()[name] = value  # 只走一次这条路
    return value


def __dir__() -> list[str]:
    """Keep tab-completion and `dir()` working despite the lazy exports.

    Returns:
        The package's public names.
    """
    return sorted(set(globals()) | set(__all__))


__version__ = "0.3.0"  # x-release-please-version
