SPHINXOPTS    ?=
SPHINXBUILD   ?= /home/amanu/miniconda3/bin/sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean html livehtml

help:
	@$(SPHINXBUILD) -M help $(SOURCEDIR) $(BUILDDIR) $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. Open $(BUILDDIR)/html/index.html"

livehtml:
	sphinx-autobuild $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS) $(O)
