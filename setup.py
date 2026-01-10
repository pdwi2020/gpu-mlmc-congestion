"""
GPU-Accelerated MLMC Network Modeling
Setup configuration for package installation
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="mlmc-network",
    version="0.1.0",
    author="Paritosh Dwivedi",
    description="GPU-Accelerated Multilevel Monte Carlo for Network Propagation and Congestion Modeling",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/GPU-Acc-Net-Prop-Congestion-Multi-Monte-Carlo",

    packages=find_packages(where="src"),
    package_dir={"": "src"},

    python_requires=">=3.10",

    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "networkx>=3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "pycuda>=2022.2",
        "numba>=0.57.0",
        "jupyter>=1.0.0",
        "pytest>=7.3.0",
        "tqdm>=4.65.0",
        "h5py>=3.8.0",
    ],

    extras_require={
        "dev": [
            "pytest-cov>=4.0.0",
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "pcap": [
            "scapy>=2.5.0",  # For MAWI PCAP processing
        ],
    },

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],

    entry_points={
        "console_scripts": [
            "mlmc-network=cli:main",  # Will create later
        ],
    },
)
