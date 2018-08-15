from setuptools import setup, find_packages


def find_scripts():
    scripts = ['bin/write_recipe.py']
    return scripts


setup(
    name='white_matter',
    version='0.91',
    install_requires=[],
    packages=find_packages(),
    include_package_data=True,
    scripts=find_scripts(),
    author='Michael Reimann',
    author_email='michael.reimann@epfl.ch',
    description='Analyze white matter connectivity in the mouse brain and write a generative recipe',
    license='Restricted',
    keywords=('neuroscience',
              'brain',
              'white matter',
              'yaml',
              'modelling'),
    url='http://bluebrain.epfl.ch',
    classifiers=['Development Status :: 3 - Alpha',
                 'Environment :: Console',
                 'License :: Proprietary',
                 'Operating System :: POSIX',
                 'Topic :: Utilities',
                 ],
)
