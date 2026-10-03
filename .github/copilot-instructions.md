## Adding or modifying code

- Before running ANY python code/script, ALWAYS use `conda activate venvnumpesdev`, except when asked explicitly to do otherwise (.e.g, use 'base' environment `venvnumpes` instead)
- Before making code changes, always run either the accompanying test suite or all tests using `tox -e full`
- After making changing, always run tests again to verify whether the implementations passes the prior failing test cases or does not break old test cases
