from setuptools import setup

setup(
    name='kasafranse',
    version='0.1',
    description='A library for Machine translation of Ghanaian language Twi',
    author='Gyasi, Frederick',
    author_email='gyasifred@gmail.com',
    license='MIT',
    install_requires=['transformers==4.24.0','sentencepiece==0.1.97', 'evaluate==0.3.0', 'googletrans==3.1.0a0',
                      'accelerate==0.14.0', 'sacrebleu==2.3.1', 'nltk==3.7', 'sacremoses==0.0.53', 'pandas==1.5.3'],
    packages=['kasafranse'],
    include_package_data=True
)
