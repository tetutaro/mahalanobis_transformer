# -*- coding: utf-8 -*-
from setuptools import setup

packages = ['mahalanobis_transformer']

package_data = {'': ['*']}

install_requires = [
    'numpy>=1.23.5,<2.0.0',
    'scikit-learn>=1.2.0,<2.0.0',
]

setup_kwargs = {
    'name': 'mahalanobis-transformer',
    'version': '0.1.0',
    'description': '',
    'long_description': None,
    'author': 'Tetsutaro Maruyama',
    'author_email': 'tetsutaro.maruyama@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9,<4.0',
}


setup(**setup_kwargs)
