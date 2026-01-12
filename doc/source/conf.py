# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'NumPES'
copyright = '2026, Bart Wolleswinkel'
author = 'Bart Wolleswinkel'

release = '0.1.0'
version = release

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_immaterial'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_immaterial'

html_theme_options = {
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "orange",
            "accent": "orange",
            "toggle": {
                "icon": "material/brightness-7",
                "name": "Switch to dark mode"
            }
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "orange",
            "accent": "orange",
            "toggle": {
                "icon": "material/brightness-4",
                "name": "Switch to light mode"
            }
        }
    ],
    "color": {
        "primary": "#FFA900"
    }
}

# -- Options for EPUB output
epub_show_urls = 'footnote'
