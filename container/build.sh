#!/usr/bin/env bash
set -e

TAG=
RUN_PREFIX=
PLATFORM=linux/amd64

# Get short commit hash
commit_id=$(git rev-parse --short HEAD)

# if COMMIT_ID matches a TAG use that
current_tag=$(git describe --tags --exact-match 2>/dev/null | sed 's/^v//') || true

# Get latest TAG and add COMMIT_ID for dev
latest_tag=$(git describe --tags --abbrev=0 "$(git rev-list --tags --max-count=1 main)" | sed 's/^v//') || true
if [[ -z ${latest_tag} ]]; then
    latest_tag="0.0.1"
    echo "No git release tag found, setting to unknown version: ${latest_tag}"
fi

# Use tag if available, otherwise use latest_tag.dev.commit_id
VERSION=v${current_tag:-$latest_tag.dev.$commit_id}
PYTHON_PACKAGE_VERSION=${current_tag:-$latest_tag.dev+$commit_id}

# Define frameworks array
declare -A FRAMEWORKS=(["VLLM"]=1 ["TENSORRTLLM"]=2 ["NONE"]=3)
DEFAULT_FRAMEWORK=VLLM

SOURCE_DIR=$(dirname "$(readlink -f "$0")")
DOCKERFILE=${SOURCE_DIR}/Dockerfile
BUILD_CONTEXT=$(dirname "$(readlink -f "$SOURCE_DIR")")

# Default base images
TENSORRTLLM_BASE_IMAGE=tensorrt_llm/release
TENSORRTLLM_BASE_IMAGE_TAG=latest
TENSORRTLLM_PIP_WHEEL_PATH=""

VLLM_BASE_IMAGE="nvcr.io/nvidia/pytorch"
VLLM_BASE_IMAGE_TAG="25.03-py3"

NONE_BASE_IMAGE="ubuntu"
NONE_BASE_IMAGE_TAG="24.04"

NIXL_COMMIT=8c4dcc1399c951632b6083303ce2e95dc7dcc7b9
NIXL_REPO=piotrm-nvidia/nixl

get_options() {
    while :; do
        case $1 in
            -h | -\? | --help)
                show_help
                exit
                ;;
            --platform)
                if [ "$2" ]; then
                    PLATFORM=$2
                    shift
                else
                    missing_requirement "$1"
                fi
                ;;
            --framework)
                if [ "$2" ]; then
                    FRAMEWORK=$2
                    shift
                else
                    missing_requirement "$1"
                fi
                ;;
            --tensorrtllm-pip-wheel-path)
                if [ "$2" ]; then
                    TENSORRTLLM_PIP_WHEEL_PATH=$2
                    shift
                else
                    missing_requirement "$1"
                fi
                ;;
            --base-image)
                if [ "$2" ]; then
                    BASE_IMAGE=$2
                    shift
                else
                    missing_requirement "$1"
                fi
                ;;
            --base-image-tag)
                if [ "$2" ]; then
                    BASE_IMAGE_TAG=$2
                    shift
                else
                    missing_requirement "$1"
                fi
                ;;
            --target)
                if [ "$2" ]; then
                    TARGET=$2
                    shift
                else
                    missing_requirement "$1"
                fi
                ;;
            --build-arg)
                if [ "$2" ]; then
                    BUILD_ARGS+="--build-arg $2 "
                    shift
                else
                    missing_requirement "$1"
                fi
                ;;
            --tag)
                if [ "$2" ]; then
                    TAG="--tag $2"
                    shift
                else
                    missing_requirement "$1"
                fi
                ;;
            --dry-run)
                RUN_PREFIX="echo"
                echo ""
                echo "=============================="
                echo "DRY RUN: COMMANDS PRINTED ONLY"
                echo "=============================="
                echo ""
                ;;
            --no-cache)
                NO_CACHE=" --no-cache"
                ;;
            --cache-from)
                if [ "$2" ]; then
                    CACHE_FROM="--cache-from $2"
                    shift
                else
                    missing_requirement "$1"
                fi
                ;;
            --cache-to)
                if [ "$2" ]; then
                    CACHE_TO="--cache-to $2"
                    shift
                else
                    missing_requirement "$1"
                fi
                ;;
            --build-context)
                if [ "$2" ]; then
                    BUILD_CONTEXT_ARG="--build-context $2"
                    shift
                else
                    missing_requirement "$1"
                fi
                ;;
            --release-build)
                RELEASE_BUILD=true
                ;;
            --)
                shift
                break
                ;;
            -?*)
                error 'ERROR: Unknown option: ' "$1"
                ;;
            ?*)
                error 'ERROR: Unknown option: ' "$1"
                ;;
            *)
                break
                ;;
        esac
        shift
    done

    # Determine or default the framework
    if [ -z "$FRAMEWORK" ]; then
        FRAMEWORK=$DEFAULT_FRAMEWORK
    fi
    FRAMEWORK=${FRAMEWORK^^}
    if [[ -z "${FRAMEWORKS[$FRAMEWORK]}" ]]; then
        error "ERROR: Unknown framework: $FRAMEWORK"
    fi

    # If user hasn't specified base image or tag, fallback to default for the chosen framework
    if [ -z "$BASE_IMAGE_TAG" ]; then
        BASE_IMAGE_TAG_VAR=${FRAMEWORK}_BASE_IMAGE_TAG
        BASE_IMAGE_TAG=${!BASE_IMAGE_TAG_VAR}
    fi
    if [ -z "$BASE_IMAGE" ]; then
        BASE_IMAGE_VAR=${FRAMEWORK}_BASE_IMAGE
        BASE_IMAGE=${!BASE_IMAGE_VAR}
    fi
    if [ -z "$BASE_IMAGE" ]; then
        error "ERROR: Framework $FRAMEWORK without BASE_IMAGE"
    fi

    # If user requested ARM64, override to ARM defaults + pass ARCH build-args
    if [[ "$PLATFORM" == *"linux/arm64"* ]]; then
        # If user hasn't explicitly set base, we do our known ARM defaults
        if [[ -z "$BASE_IMAGE" ]]; then
            BASE_IMAGE="nvcr.io/nvidia/pytorch"
        fi
        if [[ -z "$BASE_IMAGE_TAG" ]]; then
            BASE_IMAGE_TAG="25.03-py3"
        fi
        BUILD_ARGS+=" --build-arg ARCH=arm64 --build-arg ARCH_ALT=aarch64 "
    fi

    # If no user-specified tag, we create a default tag
    if [ -z "$TAG" ]; then
        TAG="--tag dynamo:${VERSION}-${FRAMEWORK,,}"
        if [ -n "${TARGET}" ]; then
            TAG="${TAG}-${TARGET}"
        fi
    fi

    # Convert platform to Docker CLI style if present
    if [ -n "$PLATFORM" ]; then
        PLATFORM="--platform ${PLATFORM}"
    fi

    # If a target is provided, pass it to Docker build, else default "dev"
    if [ -n "$TARGET" ]; then
        TARGET_STR="--target ${TARGET}"
    else
        TARGET_STR="--target dev"
    fi
}


show_image_options() {
    echo ""
    echo "Building Dynamo Image: '${TAG}'"
    echo ""
    echo "   Base: '${BASE_IMAGE}'"
    echo "   Base_Image_Tag: '${BASE_IMAGE_TAG}'"
    if [[ $FRAMEWORK == "TENSORRTLLM" ]]; then
        echo "   Tensorrtllm_Pip_Wheel_Path: '${TENSORRTLLM_PIP_WHEEL_PATH}'"
    fi
    echo "   Build Context: '${BUILD_CONTEXT}'"
    echo "   Build Arguments: '${BUILD_ARGS}'"
    echo "   Framework: '${FRAMEWORK}'"
    echo ""
}

show_help() {
    echo "usage: build.sh"
    echo "  [--base base image]"
    echo "  [--base-image-tag base image tag]"
    echo "  [--platform <platform> (e.g. linux/amd64, linux/arm64)]"
    echo "  [--framework one of ${!FRAMEWORKS[*]}]"
    echo "  [--tensorrtllm-pip-wheel-path path to tensorrtllm pip wheel]"
    echo "  [--build-arg additional build args to pass to docker build]"
    echo "  [--cache-from <cache>]"
    echo "  [--cache-to <cache>]"
    echo "  [--tag <tag> for the built image]"
    echo "  [--no-cache] disable docker build cache"
    echo "  [--dry-run] only print commands"
    echo "  [--build-context name=path to add build context]"
    echo "  [--release-build] set build to release mode"
    exit 0
}

missing_requirement() {
    error "ERROR: $1 requires an argument."
}

error() {
    printf '%s %s\n' "$1" "$2" >&2
    exit 1
}

get_options "$@"

# Choose Dockerfile based on framework
if [[ $FRAMEWORK == "VLLM" ]]; then
    DOCKERFILE=${SOURCE_DIR}/Dockerfile.vllm
elif [[ $FRAMEWORK == "TENSORRTLLM" ]]; then
    DOCKERFILE=${SOURCE_DIR}/Dockerfile.tensorrt_llm
elif [[ $FRAMEWORK == "NONE" ]]; then
    DOCKERFILE=${SOURCE_DIR}/Dockerfile.none
fi

# Possibly clone NIXL if needed for VLLM
if [[ $FRAMEWORK == "VLLM" ]]; then
    NIXL_DIR="/tmp/nixl/nixl_src"
    if [ -d "$NIXL_DIR" ]; then
        echo "Warning: $NIXL_DIR already exists, skipping clone"
    else
        if [ -n "${GITHUB_TOKEN}" ]; then
            git clone "https://oauth2:${GITHUB_TOKEN}@github.com/${NIXL_REPO}" "$NIXL_DIR"
        else
            # Try HTTPS first with credential prompting disabled
            if ! GIT_TERMINAL_PROMPT=0 git clone https://github.com/${NIXL_REPO} "$NIXL_DIR"; then
                echo "HTTPS clone failed, falling back to SSH..."
                git clone git@github.com:${NIXL_REPO} "$NIXL_DIR"
            fi
        fi
    fi

    cd "$NIXL_DIR" || exit
    if ! git checkout ${NIXL_COMMIT}; then
        echo "ERROR: Failed to checkout NIXL commit ${NIXL_COMMIT}. The cached directory may be out of date."
        echo "Please delete $NIXL_DIR and re-run the build script."
        exit 1
    fi

    BUILD_CONTEXT_ARG+=" --build-context nixl=$NIXL_DIR"
    BUILD_ARGS+=" --build-arg NIXL_COMMIT=${NIXL_COMMIT} "
fi

# local-dev target passes UID/GID
if [[ $TARGET == "local-dev" ]]; then
    BUILD_ARGS+=" --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) "
fi

# Build dev image
BUILD_ARGS+=" --build-arg BASE_IMAGE=$BASE_IMAGE"
BUILD_ARGS+=" --build-arg BASE_IMAGE_TAG=$BASE_IMAGE_TAG"
BUILD_ARGS+=" --build-arg FRAMEWORK=$FRAMEWORK"
BUILD_ARGS+=" --build-arg ${FRAMEWORK}_FRAMEWORK=1"
BUILD_ARGS+=" --build-arg VERSION=$VERSION"
BUILD_ARGS+=" --build-arg PYTHON_PACKAGE_VERSION=$PYTHON_PACKAGE_VERSION"

# Forward credentials / tokens if present
if [ -n "${GITHUB_TOKEN}" ]; then
    BUILD_ARGS+=" --build-arg GITHUB_TOKEN=${GITHUB_TOKEN} "
fi
if [ -n "${GITLAB_TOKEN}" ]; then
    BUILD_ARGS+=" --build-arg GITLAB_TOKEN=${GITLAB_TOKEN} "
fi
if [ -n "${HF_TOKEN}" ]; then
    BUILD_ARGS+=" --build-arg HF_TOKEN=${HF_TOKEN} "
fi

# Release build
if [  ! -z ${RELEASE_BUILD} ]; then
    echo "Performing a release build!"
    BUILD_ARGS+=" --build-arg RELEASE_BUILD=${RELEASE_BUILD} "
fi

# If using TensorRTLLM, pass optional wheel path
if [[ $FRAMEWORK == "TENSORRTLLM" ]]; then
    if [ -n "${TENSORRTLLM_PIP_WHEEL_PATH}" ]; then
        BUILD_ARGS+=" --build-arg TENSORRTLLM_PIP_WHEEL_PATH=${TENSORRTLLM_PIP_WHEEL_PATH} "
    fi
fi

LATEST_TAG="--tag dynamo:latest-${FRAMEWORK,,}"
if [ -n "${TARGET}" ]; then
    LATEST_TAG="${LATEST_TAG}-${TARGET}"
fi

show_image_options

if [ -z "$RUN_PREFIX" ]; then
    set -x
fi

# For TENSORRTLLM, confirm the base image is available
if [[ $FRAMEWORK == "TENSORRTLLM" ]]; then
    if docker inspect --type=image "$BASE_IMAGE:$BASE_IMAGE_TAG" > /dev/null 2>&1; then
        echo "Image '$BASE_IMAGE:$BASE_IMAGE_TAG' is found."
    else
        echo "Image '$BASE_IMAGE:$BASE_IMAGE_TAG' is not found." >&2
        echo "Please build the TensorRT-LLM base image first. Run ./build_trtllm_base_image.sh" >&2
        echo "or use --base-image and --base-image-tag to point to an existing TensorRT-LLM base image." >&2
        exit 1
    fi
fi

$RUN_PREFIX docker build \
    -f "$DOCKERFILE" \
    $TARGET_STR \
    $PLATFORM \
    $BUILD_ARGS \
    $CACHE_FROM \
    $CACHE_TO \
    $TAG \
    $LATEST_TAG \
    $BUILD_CONTEXT_ARG \
    "$BUILD_CONTEXT" \
    $NO_CACHE

{ set +x; } 2>/dev/null
if [ -z "$RUN_PREFIX" ]; then
    set -x
fi
{ set +x; } 2>/dev/null
