from setuptools import setup

setup(
   name='flow',
   version='0.0.1',
   author='Yuval Boss',
   author_email='yuval@cs.washington.edu',
   packages=['flow', 'flow.extract', 'S3Cache.S3Cache'],
   scripts=[],
   url='git@github.com:readicculus/sealnet-mlflow.git',
   license='LICENSE.txt',
   description='Package for creating machine learning datasets from the noaadb database',
   long_description=open('README.md').read(),
   install_requires=[
       "sqlalchemy >= 1.3.13",
       "psycopg2",
   ],
)