from setuptools import setup

setup(
    name='learn-it-fast',
    version='0.1.0',
    packages=['learn_it_fast'],
    package_dir={'learn_it_fast': 'learn_it_fast'},
    license='MIT',
    description='Easy transfer learning tool implemented in Pytorch',
    long_description=open('README.md').read(),
    install_requires=['torch==0.3.1', 'torchvision==0.2.0'],
    url='https://github.com/limyunkai19/learn-it-fast',
    author='Lim Yun Kai',
    author_email='yunkai96@hotmail.com'
)
