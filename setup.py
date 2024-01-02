from setuptools import setup, find_packages

setup(
    name='coxph_test',
    version='0.1',
    packages=find_packages(),
    description='A wrapper function that allows serial generation of cox proportional hazard models and testing.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Kristian Unger',
    author_email='kristian.unger@med.uni-muenchen.de',
    url='https://github.com/kristianunger/coxph_test/',
    install_requires=['pandas', 'numpy', 'lifelines', 'scipy'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)