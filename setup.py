from setuptools import setup

setup(name='BCM',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description='Baryon Correction Model',
      author='Max Edward Lee',
      author_email='mel2260@columbia.edu',
      license='GNU GPLv3',
      packages=['BCM'],
      install_requires=['numpy', 'nbodykit', 'dask[array]', 'fastpm','mpi4py','cython', 'scipy'],
      )

