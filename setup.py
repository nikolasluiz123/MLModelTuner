from setuptools import setup, find_packages

setup(
    name='MLModelTunner',
    use_scm_version=True,
    setup_requires=['setuptools-scm'],
    description='Biblioteca para auxiliar na busca do melhor modelo de machine learning.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Nikolas Luiz Schmitt',
    author_email='nikolas.luiz.schmitt@gmail.com',
    url='https://github.com/nikolasluiz123/MLModelTunner',
    packages=find_packages(),
    install_requires=[
        'contourpy', 'cycler', 'fonttools', 'joblib', 'kiwisolver', 'matplotlib',
        'numpy', 'packaging', 'pandas', 'pillow', 'pyaml', 'pyparsing', 'python-dateutil',
        'pytz', 'PyYAML', 'scikit-learn', 'scikit-optimize', 'scipy', 'seaborn', 'six',
        'tabulate', 'threadpoolctl', 'tzdata', 'xgboost', 'keras', 'keras-tuner'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
