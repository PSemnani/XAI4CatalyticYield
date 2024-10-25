from setuptools import setup, find_packages

setup(
    name="mlxai4cat",
    version="0.1.0",
    author="Parastoo Semnani",
    author_email="parastoo.semnani@outlook.com",
    description="A machine learning and explainable AI framework for catalyst discovery",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/PSemnani/XAI4CatalyticYield",
    packages=find_packages(),  # Look for packages directly in 'src'
    install_requires=[
        "imbalanced-learn>=0.12.2",
        "jupyter>=1.0.0",
        "jupyterlab>=4.1.6",
        "matplotlib>=3.8.4",
        "notebook>=7.1.3",
        "numpy>=1.26.4",
        "pandas>=2.2.2",
        "scikit-learn>=1.4.2",
        "scipy>=1.13.0",
        "seaborn>=0.13.2",
        "tabulate>=0.9.0",
        "torch>=2.1",
        "plotly>=5.24.1",
        "scikit-optimize>=0.10.2",
        "xgboost>=2.1.1",
        "tqdm>=4.65.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    include_package_data=True,  # To include non-Python files specified in MANIFEST.in
)
