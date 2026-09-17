"""Create plots from pytest-benchmark JSON output"""

# FROM: GitHub Copilot ChatGPT-5.6 Luna | 2026/10/17[untested/unverified]

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any


def _collect_plot_data(results: list[dict[str, Any]]) -> dict[str, dict[tuple[int, str], list[float]]]:
	plot_data: dict[str, dict[tuple[int, str], list[float]]] = {}
	for result in results:
		parts = result['fullname'].split('::')
		method = parts[-1].split('[', maxsplit=1)[0]
		if not method.startswith('time_'):
			continue
		benchmark_name = f'{parts[-2]}.{method}' if len(parts) > 2 else method
		params = result['params']
		category_names = [name for name in params if name != 'n']
		if 'n' not in params or len(category_names) != 1:
			continue
		category = str(params[category_names[0]])
		data = result['stats'].get('data', [])
		plot_data.setdefault(benchmark_name, {}).setdefault((int(params['n']), category), []).extend(
			value * 1000 for value in data if value > 0
		)
	return plot_data


def _show_plot(method: str, cases: dict[tuple[int, str], list[float]], scale: str) -> object:
	import plotly.graph_objects as go

	n_values = sorted({n for n, _ in cases})
	categories = list(dict.fromkeys(category for _, category in cases))
	figure = go.Figure()
	for category in categories:
		x_values = []
		y_values = []
		for n in n_values:
			values = cases.get((n, category), [])
			x_values.extend([str(n)] * len(values))
			y_values.extend(values)
		if y_values:
			figure.add_trace(go.Box(
				x=x_values,
				y=y_values,
				name=category,
			))

	max_value = max(value for values in cases.values() for value in values)
	min_value = min(value for values in cases.values() for value in values)
	log_tick_values = [2 ** exponent for exponent in range(-10, 21) if min_value / 2 <= 2 ** exponent <= max_value * 2]
	log_axis = {
		'type': 'log',
		'dtick': 1,
		'tickmode': 'array',
		'tickvals': log_tick_values,
		'ticktext': [f'{value:g}' for value in log_tick_values],
	}
	linear_axis = {'type': 'linear', 'tickformat': '~g'}
	initial_axis = log_axis if scale == 'log2' else linear_axis
	figure.update_layout(
		title={'text': method, 'x': 0.5, 'xanchor': 'center', 'font': {'family': 'Menlo'}},
		xaxis={'title': 'n', 'categoryorder': 'array', 'categoryarray': [str(n) for n in n_values]},
		yaxis={'title': 'Time (ms)', **initial_axis},
		boxmode='group',
		legend={'title': {'text': 'Parameterization'}},
		template='plotly_white',
		updatemenus=[{
			'type': 'buttons',
			'direction': 'right',
			'x': 0,
			'y': 1.15,
			'buttons': [
				{'label': 'Linear', 'method': 'relayout', 'args': [{'yaxis': {'title': 'Time (ms)', **linear_axis}}]},
				{'label': 'Log2', 'method': 'relayout', 'args': [{'yaxis': {'title': 'Time (ms)', **log_axis}}]},
			],
		}],
	)
	for seconds in (0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
		milliseconds = seconds * 1000
		if max_value * 1.5 <= milliseconds:
			continue
		figure.add_hline(y=milliseconds, line={'color': 'gray', 'dash': 'dash', 'width': 1})
		figure.add_annotation(
			x=0,
			xref='paper',
			y=milliseconds,
			yref='y',
			text=f'{seconds:g} sec',
			showarrow=False,
			xanchor='left',
			yanchor='bottom',
			font={'color': 'gray'},
		)
	return figure


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--plot', action='store_true', help='Create benchmark box plots')
	parser.add_argument('--plot-scale', choices=('linear', 'log2'), default='linear')
	parser.add_argument('--input', type=Path, default=Path('.benchmarks/latest.json'))
	parser.add_argument('--no-show', action='store_true', help='Close plots without opening a window')
	posargs = [argument for argument in sys.argv[1:] if argument]
	if len(posargs) == 1 and ' ' in posargs[0]:
		posargs = shlex.split(posargs[0])
	args = parser.parse_args(posargs)
	if not args.plot:
		return
	results = json.loads(args.input.read_text())['benchmarks']
	figures = [_show_plot(method, cases, args.plot_scale) for method, cases in _collect_plot_data(results).items()]
	if not args.no_show:
		for figure in figures:
			figure.show()


if __name__ == '__main__':
	main()
