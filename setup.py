from setuptools import setup, find_packages

setup(
    name='MLModelTunner',
    version='0.1.0',
    description='Biblioteca para auxiliar na busca do melhor modelo de machine learning.',
    # long_description=open('README.md').read(),# Descrição detalhada (exibe README.md)
    long_description_content_type='text/markdown',
    author='Nikolas Luiz Schmitt',                        # Nome do autor
    author_email='nikolas.luiz.schmitt@gmail.com',      # Email do autor
    url='https://github.com/nikolasluiz123/MLModelTunner',  # URL do projeto
    packages=find_packages(),                 # Inclui todos os pacotes do projeto
    install_requires=[
        'contourpy', 'cycler', 'fonttools', 'joblib', 'kiwisolver', 'matplotlib',
        'numpy', 'packaging', 'pandas', 'pillow', 'pyaml', 'pyparsing', 'python-dateutil',
        'pytz', 'PyYAML', 'scikit-learn', 'scikit-optimize', 'scipy', 'seaborn', 'six',
        'tabulate', 'threadpoolctl', 'tzdata', 'xgboost'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)
