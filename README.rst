Recursive Circular Seriation
-------------------------------

|PyPI download month| |PyPI version fury.io| |GitHub license| |Passing|

Python implementation of the "Recursive Seriation" algorithm for reordering strict circular Robisonian matrices presented in Armstrong, S., Guzmán, C., & Sing Long, C. A. (2021). An optimal algorithm for strict circular seriation. SIAM Journal on Mathematics of Data Science, 3(4), 1223-1250.

See:

- ArXiv_
- GoogleScholar_
- SIMODS_

Installation
------------



The package can be installed from PyPI using pip

.. code-block:: console

    pip install recursiveseriation


or from source using poetry

.. code-block:: console

    git clone https://github.com/stgoa/recursiveseriation.git
    cd recursiveseriation
    python -m venv .venv
    poetry install

Usage
------------

The package provides a single class `RecursiveSeriation` that takes a strict circular Robisonian dissimilarity and returns the permutation that correctly orders the elements in the matrix in a circular way.

.. code-block:: python

    from recursiveseriation.solver.seriation import RecursiveSeriation

    # example of a strict circular Robisonian dissimilarity (matrix)
    R = [
            [0, 1, 3, 5, 6, 7, 7, 6, 5, 4, 3],
            [1, 0, 2, 4, 5, 6, 7, 7, 6, 5, 4],
            [3, 2, 0, 1, 4, 5, 6, 7, 7, 6, 5],
            [5, 4, 1, 0, 1, 4, 5, 6, 7, 7, 6],
            [6, 5, 4, 1, 0, 1, 4, 5, 6, 7, 7],
            [7, 6, 5, 4, 1, 0, 3, 4, 5, 6, 7],
            [7, 7, 6, 5, 4, 3, 0, 1, 4, 5, 6],
            [6, 7, 7, 6, 5, 4, 1, 0, 2, 4, 5],
            [5, 6, 7, 7, 6, 5, 4, 2, 0, 1, 4],
            [4, 5, 6, 7, 7, 6, 5, 4, 1, 0, 1],
            [3, 4, 5, 6, 7, 7, 6, 5, 4, 1, 0],
        ]

    # number of elements 
    n = len(R)

    # dissimilarity function over
    # {0, ..., n-1}^2 -> [0, inf)
    dissimilarity = lambda i, j: R[i][j]

    rs = RecursiveSeriation(
        dissimilarity=dissimilarity,
        n=n,
    )
    order = rs.sort()
    # we obtain the identity permutation up to cyclic permutations and reversals (the elements are already ordered)
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

For examples take a look at the notebooks:

- example_points_in_circle.ipynb_
- minimal_example.ipynb_ 
- performance.ipynb_

.. _ArXiv: https://arxiv.org/abs/2106.05944
.. _GoogleScholar: https://scholar.google.com/citations?view_op=view_citation&hl=en&user=_VV7RLwAAAAJ&citation_for_view=_VV7RLwAAAAJ:u5HHmVD_uO8C
.. _SIMODS: https://epubs.siam.org/doi/abs/10.1137/21M139356X
.. _minimal_example.ipynb: examples/minimal_example.ipynb
.. _example_points_in_circle.ipynb: examples/example_points_in_circle.ipynb
.. _performance.ipynb: examples/performance.ipynb
.. |PyPI version fury.io| image:: https://badge.fury.io/py/recursiveseriation.svg
   :target: https://pypi.python.org/pypi/recursiveseriation/
.. |PyPI download month| image:: https://img.shields.io/pypi/dm/recursiveseriation.svg
   :target: https://pypi.python.org/pypi/recursiveseriation/
.. |GitHub license| image:: https://img.shields.io/github/license/stgoa/recursiveseriation.svg
   :target: https://github.com/stgoa//recursiveseriation/blob/main/LICENSE
.. |Passing| image:: https://github.com/stgoa/recursiveseriation/actions/workflows/ci.yml/badge.svg?branch=main
   :target: https://github.com/stgoa/recursiveseriation/actions/workflows/ci.yml
