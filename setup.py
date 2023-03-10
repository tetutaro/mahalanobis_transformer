# -*- coding: utf-8 -*-
from setuptools import setup

packages = ['mahalanobis_transformer']

package_data = {'': ['*']}

install_requires = [
    'numpy>=1.23.5,<2.0.0',
    'scikit-learn>=1.2.0,<2.0.0',
]

entry_points = {'console_scripts': ['test = scripts:test']}

setup_kwargs = {
    'name': 'mahalanobis-transformer',
    'version': '0.1.2',
    'description': (
        "The transformer that transforms data "
        "so to squared norm of transformed data "
        "becomes Mahalanobis' distance"
    ),
    'long_description': None,
    'author': 'Tetsutaro Maruyama',
    'author_email': 'tetsutaro.maruyama@gmail.com',
    'maintainer': 'Tetsutaro Maruyama',
    'maintainer_email': 'tetsutaro.maruyama@gmail.com',
    'url': 'https://github.com/tetutaro/mahalanobis_transformer',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.9,<4.0',
}


setup(**setup_kwargs)
