"""Setup module for performing benchmarks"""

# FROM: GitHub Copilot ChatGPT-5.6 Luna | 2026/10/17[untested/unverified]

import pytest
from pytest_benchmark import table


table.NUMBER_FMT = '{0:,.2f}'
table.ALIGNED_NUMBER_FMT = '{0:>{1},.2f}{2:<{3}}'

_display = table.TableResults.display


class _SpaceGroupedReporter:
	def __init__(self, reporter: object) -> None:
		self._reporter = reporter

	def __getattr__(self, name: str) -> object:
		return getattr(self._reporter, name)

	def write(self, content: str, **markup: bool) -> None:
		self._reporter.write(content.replace(',', ' '), **markup)

	def write_line(self, content: str, **markup: bool) -> None:
		self._reporter.write_line(content.replace(',', ' '), **markup)


def _display_with_space_grouping(self: table.TableResults, reporter: object, *args: object, **kwargs: object) -> None:
	_display(self, _SpaceGroupedReporter(reporter), *args, **kwargs)


table.TableResults.display = _display_with_space_grouping


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
	"""Mark time-based benchmark methods for pytest-benchmark"""

	for item in items:
		if item.name.rsplit('.', maxsplit=1)[-1].startswith('time_'):
			item.fixturenames.append('benchmark')


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
	"""Run time-based benchmark methods through pytest-benchmark"""

	if not pyfuncitem.name.rsplit('.', maxsplit=1)[-1].startswith('time_'):
		return None

	benchmark = pyfuncitem._request.getfixturevalue('benchmark')
	parameters = getattr(pyfuncitem, 'callspec', None)
	benchmark(pyfuncitem.obj, **parameters.params)
	return True
