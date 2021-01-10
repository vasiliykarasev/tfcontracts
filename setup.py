from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='tfcontrats',
    version='0.1.0',
    description='Contract-based programming for tensorflow',
    long_description=readme,
    author='vasiliykarasev',
    author_email='karasev00@gmail.com',
    url='https://github.com/vasiliykarasev/tfcontracts',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

