Recursive Circular Seriation
-------------------------------

Python implementation of the "Recursive Seriation" algorithm for reordering strict circular Robisonian matrices presented in Armstrong, S., Guzmán, C., & Sing Long, C. A. (2021). An optimal algorithm for strict circular seriation. SIAM Journal on Mathematics of Data Science, 3(4), 1223-1250.

See:

- ArXiv_
- GoogleScholar_
- SIMODS_

For examples take a look at the notebooks:

- example_points_in_circle.ipynb_
- minimal_example.ipynb_ 
- performance.ipynb_

Installation
------------

The package can be installed using poetry

.. code-block:: console

    $ git clone https://github.com/stgoa/recursiveseriation.git
    $ cd recursiveseriation
    $ python -m venv .venv
    $ poetry install






.. _ArXiv: https://arxiv.org/abs/2106.05944
.. _GoogleScholar: https://scholar.google.com/citations?view_op=view_citation&hl=en&user=_VV7RLwAAAAJ&citation_for_view=_VV7RLwAAAAJ:u5HHmVD_uO8C
.. _SIMODS: https://epubs.siam.org/doi/abs/10.1137/21M139356X
.. _minimal_example.ipynb: examples/minimal_example.ipynb
.. _example_points_in_circle.ipynb: examples/example_points_in_circle.ipynb
.. _performance.ipynb: examples/performance.ipynb