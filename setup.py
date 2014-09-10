
import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='pysolve',
    version='0.1.2',
    packages=find_packages(),
    install_requires=['sympy', 'numpy'],
    license='MIT',
    author='Kenn Takara',
    author_email='kenn.takara@outlook.com',
    classifiers={
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Mathematics',
    },
    url='https://github.com/kennt/pylinsolve',
    long_description=read('README.rst'),
)
