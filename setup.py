from distutils.core import setup

setup(name='keyVD',
      version='1.0.0',
      description='dictionary-based keyword generation',
      author='saber zahhar',
      author_email='zahhar.saber@gmail.com',
      license='gnu',
      url="https://github.com/saberzahhar/keyVD",
      install_requires=[
          'nltk',
          'pandas',
          'numpy',
          'scipy',
          'scikit-learn'
      ]
     )
