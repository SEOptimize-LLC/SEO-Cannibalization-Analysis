from setuptools import setup, find_packages

setup(
    name='seo-cannibalization-tool',
    version='2.0.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'openpyxl>=3.1.0',
        'pyyaml>=6.0',
        'click>=8.1.0',
        'tqdm>=4.65.0',
    ],
    entry_points={
        'console_scripts': [
            'seo-analyze=main:main',
        ],
    },
    author='SEOptimize LLC',
    description='Advanced SEO Cannibalization Analysis Tool',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/SEOptimize-LLC/SEO-Cannibalization-Analysis',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Internet :: WWW/HTTP :: Site Management',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
)
