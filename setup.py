from setuptools import setup, find_packages

setup(
    name="perception_chains",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "qwen-vl-utils>=0.0.8",
        "Pillow>=10.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "tqdm",
        "pyyaml",
        "einops",
    ],
)
