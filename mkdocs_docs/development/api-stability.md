# API stability

fincore 0.5 is a breaking Beta release. The stable direction is the canonical
domain layout: each public operation has one owning module, structured input
contracts, and executable domain tests.

The following are explicitly **not** public APIs in 0.5:

- upstream-shaped Empyrical, Pyfolio, and Alphalens import paths;
- root-level metric re-exports and façade classes;
- dynamic aliases, compatibility extras, and process-global plugin registries;
- private modules prefixed with `_`.

Before a 1.0 release, domain APIs may evolve where tests and migration notes
record the change. New contributions should add direct leaf APIs rather than
another cross-domain convenience façade.
