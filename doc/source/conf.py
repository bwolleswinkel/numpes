# Configuration file for the Sphinx documentation builder.
import subprocess
import re
import os

# -- Project information

project = 'NumPES'
copyright = '2026, Bart Wolleswinkel'
author = 'Bart Wolleswinkel'


def get_release_version() -> str:
    """Get the latest git tag version from the repository"""
    # FROM: GitHub Copilot, Claude Sonnet 4 | 2026/01/12
    try:
        # First, fetch tags from remote to ensure we have the latest
        subprocess.run(['git', 'fetch', '--tags'], 
                      capture_output=True, cwd=os.path.abspath('../..'))
        
        # Get all tags sorted by version (semantic versioning)
        result = subprocess.run(['git', 'tag', '-l', '--sort=-version:refname'], 
                              capture_output=True, text=True, 
                              cwd=os.path.abspath('../..'))
        if result.returncode == 0:
            tags = [tag.strip() for tag in result.stdout.split('\n') if tag.strip()]
            # Filter for version-like tags only
            version_pattern = r'^v?\d+\.\d+'
            
            for tag in tags:
                if re.match(version_pattern, tag) and 'dev' not in tag.lower():
                    # Clean up the version tag (remove 'v' prefix if present)
                    clean_version = re.sub(r'^v', '', tag)
                    return clean_version
    except Exception as e:
        print(f"Warning: Could not get git version: {e}")
    return '0.1.0'  # fallback


release = get_release_version()
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

# -- Options for EPUB output
epub_show_urls = 'footnote'
