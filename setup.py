import os
import shutil

from setuptools import setup

with open(os.path.join("rl_zoo3", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()

# Copy hyperparams files for packaging
shutil.copytree("hyperparams", os.path.join("rl_zoo3", "hyperparams"))

long_description = """
# RL-UQ-Bench: Uncertainty Quantification Benchmark for Deep Reinforcement Learning

A comprehensive evaluation framework for uncertainty quantification (UQ) methods in deep reinforcement learning agents.

See https://github.com/huangf06/rl-uq-bench
"""
install_requires = [
    "sb3_contrib>=2.7.0a0,<3.0",
    "gymnasium>=0.29.1,<1.2.0",
    "huggingface_sb3>=3.0,<4.0",
    "tqdm",
    "rich",
    "optuna>=3.0",
    "pyyaml>=5.1",
    "pytablewriter~=1.2",
    "shimmy~=2.0",
]
plots_requires = ["seaborn", "rliable~=1.2.0", "scipy~=1.10"]
test_requires = [
    # for MuJoCo envs v4:
    "mujoco>=2.3,<4",
    # install parking-env to test HER
    "highway-env>=1.10.1,<1.11.0",
]

setup(
    name="rl_zoo3",
    packages=["rl_zoo3", "rl_zoo3.plots"],
    package_data={
        "rl_zoo3": [
            "py.typed",
            "version.txt",
            "hyperparams/*.yml",
        ]
    },
    entry_points={"console_scripts": ["rl_zoo3=rl_zoo3.cli:main"]},
    install_requires=install_requires,
    extras_require={"plots": plots_requires, "tests": test_requires},
    description="Uncertainty Quantification Benchmark for Deep Reinforcement Learning",
    author="Fei Huang",
    url="https://github.com/huangf06/rl-uq-bench",
    author_email="f.huang@student.tudelft.nl",
    keywords="uncertainty-quantification reinforcement-learning machine-learning "
    "gymnasium openai stable baselines sb3 benchmark deep-learning uq",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    python_requires=">=3.9",
    # PyPI package information.
    project_urls={
        "Code": "https://github.com/huangf06/rl-uq-bench",
        "Documentation": "https://rl-baselines3-zoo.readthedocs.io/en/master/",
        "Changelog": "https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/CHANGELOG.md",
        "Stable-Baselines3": "https://github.com/DLR-RM/stable-baselines3",
        "RL-Zoo": "https://github.com/DLR-RM/rl-baselines3-zoo",
        "SBX": "https://github.com/araffin/sbx",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)

# Remove copied files after packaging
shutil.rmtree(os.path.join("rl_zoo3", "hyperparams"))
