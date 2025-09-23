from setuptools import setup, find_packages

description = """
Offical implementation of ICRA 2025:
<Robust Scene Change Detection Using Visual Foundation Models and Cross-Attention Mechanisms>
"""

setup(
    name="robust-scene-change-detection",
    version="0.1.0",
    description=description,
    author="Chad, Chun-Jung Lin",
    url="https://chadlin9596.github.io/Robust-Scene-Change-Detection/",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "einops",
        "matplotlib",
        "numpy<2.0.0",
        "pyyaml",
        "Pillow",
        "torch==2.6.0",
        "torchvision==0.21.0",
        "wandb",
        "xformers==0.0.29.post2",
    ],
)
