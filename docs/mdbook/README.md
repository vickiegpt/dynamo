## mdBook Documentation

This directory contains an mdBook build of Dynamo documentation. It complements the existing Sphinx docs in `docs/`.

### Prerequisites

- Rust toolchain and Cargo
- mdBook and link checker:
  - `cargo install mdbook mdbook-linkcheck`

### Commands

- Serve locally: `make mdbook-serve` (from `docs/`)
- Build static site: `make mdbook-build`
- Clean output: `make mdbook-clean`

The generated site will be in `docs/mdbook/book/`.

