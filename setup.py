from setuptools import setup

__VERSION__ = '0.3.0'

setup(name='h5database',
      version=__VERSION__,
      description='HDF5 Database for Images',
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      keywords=['pytable', 'hdf5', 'patch'],
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
      ],
      url='https://github.com/CielAl/H5Database',
      author='***',
      author_email='**',
      license='MIT',
      packages=['h5database'],
      install_requires=[
          'tables>=3.4.4',
          'scikit-learn>=0.20.2',
          'numpy>=1.14.5',
          'tqdm>=4.28.1',
          'lazy_property>=0.0.1',
          'pillow>=5.4.0',
          'joblib>=0.13.2',
          'scikit-image>=0.14.1',
          'opencv-python>=3.4.5.20',
      ],
      zip_safe=False,
      python_requires='>=3.6.0')
