"""
AEGIS - Advanced Enterprise Geometric Intelligence System
Setup configuration for Python package installation.

Install in development mode:
    pip install -e .

Install for production:
    pip install .
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            return [
                line.strip() 
                for line in f 
                if line.strip() and not line.startswith('#')
            ]
    return []

setup(
    name='aegis',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Institutional-grade quantitative risk management platform',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/aegis',
    
    # Package discovery
    package_dir={'': 'src/python'},
    packages=find_packages(where='src/python'),
    
    # Python version requirement
    python_requires='>=3.10',
    
    # Dependencies
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'matplotlib>=3.7.0',
        'plotly>=5.15.0',
        'POT>=0.9.0',  # Python Optimal Transport
    ],
    
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.7.0',
            'flake8>=6.1.0',
            'mypy>=1.4.0',
        ],
        'data': [
            'yfinance>=0.2.28',
            'alpha-vantage>=2.3.1',
        ],
        'ml': [
            'torch>=2.0.0',
            'transformers>=4.30.0',
        ],
        'all': [
            'yfinance>=0.2.28',
            'torch>=2.0.0',
            'transformers>=4.30.0',
        ]
    },
    
    # Entry points for CLI
    entry_points={
        'console_scripts': [
            'aegis=main:main',
        ],
    },
    
    # Metadata
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    
    keywords='quantitative-finance, portfolio-optimization, differential-geometry, risk-management, visualization',
    
    # Include non-Python files
    include_package_data=True,
    
    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/aegis/issues',
        'Documentation': 'https://github.com/yourusername/aegis/docs',
        'Source': 'https://github.com/yourusername/aegis',
    },
)
