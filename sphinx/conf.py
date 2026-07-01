import os
import sys

# Make the project importable without installing it
sys.path.insert(0, os.path.abspath(".."))

project = "CRIE-Bench"
copyright = "2025, CRIE-Bench Authors"
author = "CRIE-Bench Authors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Mock heavy C-extension and MuJoCo-dependent packages so autodoc can import
# modules that transitively depend on them without a full roco conda env.
autodoc_mock_imports = [
    "numpy",
    "mujoco",
    "dm_control",
    "open3d",
    "torch",
    "torchvision",
    "gymnasium",
    "metaworld",
    "transformers",
    "hydra",
    "omegaconf",
    "lerobot",
    "cv2",
    "PIL",
    "scipy",
    "sklearn",
    "matplotlib",
    "google",
    "openai",
    "transforms3d",
    "pydantic",
    "pyquaternion",
    "robosuite",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = []  # no _static dir needed until we add real assets
html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

myst_enable_extensions = ["colon_fence", "deflist"]
