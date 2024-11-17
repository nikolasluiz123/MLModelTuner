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
        'absl-py', 'astunparse', 'certifi', 'charset-normalizer', 'colorama', 'contourpy', 'cycler', 'flatbuffers',
        'fonttools', 'gast', 'google-pasta', 'grpcio', 'h5py', 'idna', 'joblib', 'kagglehub', 'keras', 'keras-tuner',
        'kiwisolver', 'kt-legacy', 'libclang', 'Markdown', 'markdown-it-py', 'MarkupSafe', 'matplotlib', 'mdurl',
        'ml-dtypes', 'namex', 'numpy', 'opencv-python', 'opt_einsum', 'optree', 'packaging', 'pandas', 'pillow',
        'protobuf', 'pyaml', 'Pygments', 'pyparsing', 'python-dateutil', 'pytz', 'PyYAML', 'requests', 'rich',
        'scikit-learn', 'scikit-optimize', 'scipy', 'seaborn', 'setuptools-scm', 'six', 'tabulate', 'tensorboard',
        'tensorboard-data-server', 'tensorflow', 'tensorflow-io-gcs-filesystem', 'tensorflow_intel', 'termcolor',
        'threadpoolctl', 'tomli', 'tqdm', 'typing_extensions', 'tzdata', 'urllib3', 'Werkzeug', 'wrapt', 'xgboost'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
