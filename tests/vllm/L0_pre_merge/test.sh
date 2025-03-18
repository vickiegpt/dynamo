#!/usr/bin/env bash

set -xe

TEST_NAME="$(basename "$(dirname "$(readlink -f "$0")")")"
OUTPUT_DIR=${OUTPUT_DIR:-$PWD/.tests_outputs/${TEST_NAME}}
DYNAMO_REPO_DIR=${DYNAMO_REPO_DIR:-$PWD}

# TODO: recover mypy checks

pytest \
    --md-report \
    --md-report-verbose=6 \
    --md-report-output="${OUTPUT_DIR}/standard_pytest_report.md" \
    --junitxml="${OUTPUT_DIR}/standard_pytest_report.xml" \
    -m 'pre_merge' \
    $DYNAMO_REPO_DIR