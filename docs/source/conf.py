"""Configuration file for the Sphinx documentation builder"""

from docutils import nodes
from sphinx import addnodes

project = 'NumPES'
copyright = '2026, Bart Wolleswinkel'
author = 'Bart Wolleswinkel'

release = '0.1.0'
version = release
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_design',
]

html_theme = 'furo'

html_logo = '_static/logo.svg'
html_title = 'NumPES'
html_show_sourcelink = False
# FROM: GitHub Copilot GPT-5.6 Luna | 2026/10/08 
html_theme_options = {
    'light_css_variables': {
        'color-brand-primary': '#e67e22',
        'color-brand-content': '#e67e22',
        'color-brand-visited': '#e67e22',
        'color-link': '#e67e22',
        'color-link--visited': '#e67e22',
        'color-link--hover': '#e67e22',
        'color-link--visited--hover': '#e67e22',
        'color-sidebar-item-foreground--normal': '#e67e22',
        'color-sidebar-item-foreground--current': '#e67e22',
        'color-sidebar-item-foreground--hover': '#e67e22',
    },
    'dark_css_variables': {
        'color-brand-primary': '#e67e22',
        'color-brand-content': '#e67e22',
        'color-brand-visited': '#e67e22',
        'color-link': '#e67e22',
        'color-link--visited': '#e67e22',
        'color-link--hover': '#e67e22',
        'color-link--visited--hover': '#e67e22',
        'color-sidebar-item-foreground--normal': '#e67e22',
        'color-sidebar-item-foreground--current': '#e67e22',
        'color-sidebar-item-foreground--hover': '#e67e22',
    },
    'footer_icons': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/bwolleswinkel/numpes',
            'html': (
                # FIXME: Move this to separate .svg file, but that gave me color problems before
                '<svg stroke="currentColor" fill="currentColor" '
                'stroke-width="0" viewBox="0 0 16 16">'
                '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 '
                '5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>'
                '</svg>'
            ),
        },
    ],
}

html_static_path = ['_static']
html_css_files = ['custom.css']
templates_path = ['_templates']


def _document_title(app, docname):
    """Return the title parsed from an RST document."""
    title = app.env.get_doctree(docname).next_node(nodes.title)
    return title.astext() if title is not None else docname


def _toctree_caption(app, parent, child):
    """Return the caption of the toctree containing a document."""
    doctree = app.env.get_doctree(parent)
    for toctree in doctree.findall(addnodes.toctree):
        entries = [entry for _, entry in toctree.get('entries', [])]
        if child in entries:
            return toctree.get('caption', '')
    return ''


def _toctree_entries(app, parent):
    """Return the direct document children declared by a page toctree."""
    entries = []
    for toctree in app.env.get_doctree(parent).findall(addnodes.toctree):
        for _, entry in toctree.get('entries', []):
            entries.append(entry)
    return entries


def _toctree_sections(app, parent):
    """Return each page toctree with its caption and document children."""
    sections = []
    for toctree in app.env.get_doctree(parent).findall(addnodes.toctree):
        sections.append(
            {
                'caption': toctree.get('caption', ''),
                'entries': [entry for _, entry in toctree.get('entries', [])],
            }
        )
    return sections


def _add_navigation_context(app, pagename, templatename, context, doctree):
    context['document_title'] = lambda docname: _document_title(app, docname)
    context['toctree_caption'] = lambda parent, child: _toctree_caption(
        app, parent, child
    )
    context['toctree_entries'] = lambda parent: _toctree_entries(app, parent)
    context['toctree_sections'] = lambda parent: _toctree_sections(app, parent)


def setup(app):
    app.connect('html-page-context', _add_navigation_context)
