from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="spaces-game",
    version="0.1.0",
    author="Ryan Khetlyr",
    description="Spaces Game engine for ML/RL training with Gymnasium integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/spaces-game-python",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "typing-extensions>=4.5.0",
        "gymnasium>=0.29.0",
        "click>=8.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "hypothesis>=6.80.0",
            "mypy>=1.4.0",
            "black>=23.0.0",
        ],
        "rl": [
            "stable-baselines3>=2.0.0",
            "tensorboard>=2.14.0",
        ],
    },
    package_data={
        "spaces_game": ["py.typed"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "spaces-game=spaces_game.cli:cli",
        ],
    },
)
