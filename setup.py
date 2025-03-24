from setuptools import setup

setup(
    name='mlworkflow',
    version='0.2.6',
    packages=['mlworkflow', 'mlworkflow.test', 'mlworkflow.models'],
    scripts=[],
    url='https://github.com/aitenea/ml-workflow/tree/main',
    license='MIT',
    author='AItenea Biotech',
    author_email='david.quesada@aitenea.es',
    description='Python package for training and prediction in machine learning workflows.',
    long_description=open('README.txt').read(),
    install_requires=['pandas', 'numpy', 'rdkit', 'pubchempy', 'scikit-learn', 'tqdm', 'matplotlib']
)