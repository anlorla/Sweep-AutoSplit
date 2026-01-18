"""
Sweep Auto Split - 安装脚本

推荐使用 pyproject.toml 安装:
    pip install -e .

或者使用 setup.py:
    python setup.py install
"""

from setuptools import setup, find_packages

setup(
    name="sweep_auto_split",
    version="0.2.0",
    description="基于 Pi/LeRobot 的 Sweep 动作自动切分工具",
    author="Zeno",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.4.0",
        "pyarrow>=8.0.0",
        "opencv-python>=4.5.0",
        "lerobot>=0.1.0",
    ],
    extras_require={
        "viz": [
            "matplotlib>=3.5.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "sweep-auto-split=sweep_auto_split.main:main",
        ],
    },
)
