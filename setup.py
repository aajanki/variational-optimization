import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

setup(name='variationaloptimization',
      version='0.1',
      description='Derivative-free function minimization on a binary vector domain',
      author='Antti Ajanki',
      author_email='antti.ajanki@iki.fi',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      setup_requires = ['pytest-runner'],
      tests_require=[
          'pytest',
          'numpy'
      ],
      license='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Mathematics'
      ]
)
