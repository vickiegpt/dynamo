# Dockerfile Templates

This directory contains Jinja2 templates for generating Dockerfiles. The templating system allows for modular, maintainable Dockerfile generation with shared components and standardized headers.

## Dockerfile Composition

```mermaid
flowchart TB
    %% Template Compositions (showing internal structure)
    subgraph DFJ[Dockerfile.j2]
        direction TB
        DFJ_H[_header_comment.j2]
        DFJ_BODY["main content<br/>(366 lines)<br/><br/><br/><br/><br/><br/>"]
        DFJ_E[_entrypoint.j2]
        DFJ_H --> DFJ_BODY --> DFJ_E
    end

    subgraph DVJ[Dockerfile.vllm.j2]
        direction TB
        DVJ_H[_header_comment.j2]
        DVJ_BODY["vllm content<br/>(425 lines)<br/><br/><br/><br/><br/><br/><br/>"]
        DVJ_E1[_entrypoint.j2]
        DVJ_D1[_dev_utils.j2]
        DVJ_E2[_entrypoint.j2]
        DVJ_D2[_dev_utils.j2]
        DVJ_E3[_entrypoint.j2]
        DVJ_H --> DVJ_BODY --> DVJ_E1 --> DVJ_D1 --> DVJ_E2 --> DVJ_D2 --> DVJ_E3
    end

    subgraph DSJ[Dockerfile.sglang.j2]
        direction TB
        DSJ_H[_header_comment.j2]
        DSJ_BODY["sglang content<br/>(297 lines)<br/><br/><br/><br/><br/>"]
        DSJ_E1[_entrypoint.j2]
        DSJ_D[_dev_utils.j2]
        DSJ_E2[_entrypoint.j2]
        DSJ_H --> DSJ_BODY --> DSJ_E1 --> DSJ_D --> DSJ_E2
    end

    subgraph DTJ[Dockerfile.trtllm.j2]
        direction TB
        DTJ_H[_header_comment.j2]
        DTJ_BODY["trtllm content<br/>(502 lines)<br/><br/><br/><br/><br/><br/><br/><br/>"]
        DTJ_E1[_entrypoint.j2]
        DTJ_E2[_entrypoint.j2]
        DTJ_H --> DTJ_BODY --> DTJ_E1 --> DTJ_E2
    end

    subgraph DPJ[Dockerfile.trtllm_prebuilt.j2]
        direction TB
        DPJ_H[_header_comment.j2]
        DPJ_BODY["prebuilt content<br/>(77 lines)<br/>"]
        DPJ_H --> DPJ_BODY
    end

    subgraph DDJ[Dockerfile.docs.j2]
        direction TB
        DDJ_H[_header_comment.j2]
        DDJ_BODY["docs content<br/>(41 lines)<br/>"]
        DDJ_H --> DDJ_BODY
    end

    subgraph DSWJ[Dockerfile.sglang-wideep.j2]
        direction TB
        DSWJ_H[_header_comment.j2]
        DSWJ_BODY["sglang-wideep content<br/>(131 lines)<br/><br/><br/>"]
        DSWJ_H --> DSWJ_BODY
    end

    %% Generation arrows
    DFJ ==> DF[Dockerfile]
    DVJ ==> DV[Dockerfile.vllm]
    DSJ ==> DS[Dockerfile.sglang]
    DTJ ==> DT[Dockerfile.trtllm]
    DPJ ==> DP[Dockerfile.trtllm_prebuilt]
    DDJ ==> DD[Dockerfile.docs]
    DSWJ ==> DSW[Dockerfile.sglang-wideep]

    %% Styling
    classDef templateBox fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef sharedFragment fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef mainContent fill:#fce4ec,stroke:#880e4f,stroke-width:1px
    classDef dockerfile fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px

    class DFJ,DVJ,DSJ,DTJ,DPJ,DDJ,DSWJ templateBox
    class DFJ_H,DFJ_E,DVJ_H,DVJ_D1,DVJ_E1,DVJ_D2,DVJ_E2,DVJ_E3,DSJ_H,DSJ_E1,DSJ_D,DSJ_E2,DTJ_H,DTJ_E1,DTJ_E2,DPJ_H,DDJ_H,DSWJ_H sharedFragment
    class DFJ_BODY,DVJ_BODY,DSJ_BODY,DTJ_BODY,DPJ_BODY,DDJ_BODY,DSWJ_BODY mainContent
    class DF,DV,DS,DT,DP,DD,DSW dockerfile
```

## Overview

The `gen_dockerfiles.py` script processes `.j2` template files and generates corresponding Dockerfiles in `/tmp` (or a specified output directory). It also provides comparison functionality to verify that generated files match their originals in the `container/` directory.

## Usage

```bash
# Generate all Dockerfiles and compare with originals (default output-dir: /tmp)
python3 gen_dockerfiles.py --compare-ignore-whitespaces

# Generate and perform strict comparison with detailed differences
python3 gen_dockerfiles.py --compare-strict --show-differences

# Generate to specific output directory
python3 gen_dockerfiles.py --output-dir /path/to/output

# Generate without comparison (default output-dir: /tmp)
python3 gen_dockerfiles.py
```

## Template Structure

### Main Templates
- `Dockerfile.j2` - Main Dockerfile template
- `Dockerfile.vllm.j2` - vLLM variant
- `Dockerfile.sglang.j2` - SGLang variant
- `Dockerfile.trtllm.j2` - TensorRT-LLM variant
- `Dockerfile.docs.j2` - Documentation build
- `Dockerfile.sglang-wideep.j2` - SGLang with Wideep
- `Dockerfile.trtllm_prebuilt.j2` - Prebuilt TensorRT-LLM

### Template Fragments
Template fragments (files with `._` in the name) are reusable components that are included by main templates but not generated as standalone Dockerfiles:

- `Dockerfile._header_comment.j2` - Standardized header with syntax directive and SPDX license
- `Dockerfile._dev_utils.j2` - Common development utilities installation

### Master Template (Future)
- `Dockerfile_master.j2` - Conditional template that can generate different variants based on `dockerfile_type` variable

## Template Features

### Standardized Headers
All templates include the standardized header via:
```jinja2
{% include 'Dockerfile._header_comment.j2' %}
```

This ensures consistent syntax directives and SPDX license headers across all generated Dockerfiles.

### Modular Components
Common sections are extracted into reusable fragments:
```jinja2
{% include 'Dockerfile._dev_utils.j2' %}
```

### Conditional Logic
Templates use Jinja2 conditionals for dynamic content:
```jinja2
{% if type == "dev" or type == "local-dev" %}
# Development-specific content
{% endif %}
```


## Comparison Logic

The script provides two comparison modes:

### `--compare-ignore-whitespaces` (Lenient Comparison)
- Normalizes all whitespace and removes blank lines
- Ignores template header comments (`# TEMPLATE:` lines)
- Ideal for verifying content equivalence regardless of formatting
- Uses Unix `diff` for detailed output when files differ

### `--compare-strict` (Strict Comparison)
- Preserves all whitespace differences (spaces, tabs, newlines)
- Only ignores template header comments (`# TEMPLATE:` lines)
- Detects formatting differences like missing newlines at end of files
- Useful for ensuring exact file formatting matches

### Common Features
- Both modes filter out auto-generated `# TEMPLATE:` header comments
- Provide clear success/failure indicators with colored output
- Support `--show-differences` flag for detailed diff output
- When both flags are used together, `--compare-strict` takes precedence

## File Filtering

The generator automatically skips template fragments (files containing `._` in the filename) during generation and comparison, ensuring only complete Dockerfiles are processed.

## Maintenance

To modify Dockerfiles:
1. Edit the corresponding `.j2` template file
2. Run `python3 gen_dockerfiles.py --compare-ignore-whitespaces` to verify content changes
3. Optionally run `python3 gen_dockerfiles.py --compare-strict --show-differences` to check formatting
4. Copy generated files from `/tmp` to `container/` if needed
5. Commit both template and generated files

### Comparison Workflow Recommendations
- Use `--compare-ignore-whitespaces` for content verification during development
- Use `--compare-strict --show-differences` before final commits to see exact formatting differences
- The `--show-differences` flag is especially useful with strict comparison to identify specific whitespace issues

