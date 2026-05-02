#!/usr/bin/env sh

normalize_arch() {
    value=$(printf '%s' "$1" | tr -d '\r' | awk '{$1=$1; print}')
    case "$value" in
        ""|auto|native)
            printf '\n'
            return
            ;;
    esac

    if printf '%s\n' "$value" | grep -Eq '^[0-9]+[.][0-9]+$'; then
        major=${value%.*}
        minor=${value#*.}
        printf 'compute_%s%s\n' "$major" "$minor"
        return
    fi

    if printf '%s\n' "$value" | grep -Eq '^[0-9]{2,3}$'; then
        printf 'compute_%s\n' "$value"
        return
    fi

    if printf '%s\n' "$value" | grep -Eiq '^(compute|sm)_[0-9]{2,3}$'; then
        prefix=$(printf '%s' "${value%%_*}" | tr '[:upper:]' '[:lower:]')
        number=${value#*_}
        printf '%s_%s\n' "$prefix" "$number"
        return
    fi

    printf '%s\n' "$value"
}

arch_number() {
    arch=$1
    case "$arch" in
        compute_[0-9]*|sm_[0-9]*)
            printf '%s\n' "${arch#*_}"
            ;;
        *)
            printf '%s\n' "-1"
            ;;
    esac
}

contains_number() {
    target=$1
    for arch in $supported; do
        if [ "$arch" = "$target" ]; then
            return 0
        fi
    done

    return 1
}

fallback=$(normalize_arch "${1:-compute_61}")
if [ -z "$fallback" ]; then
    fallback=compute_61
fi

env_arch=$(normalize_arch "${TENSORSHARP_CUDA_ARCH:-}")
if [ -n "$env_arch" ]; then
    printf '%s\n' "$env_arch"
    exit 0
fi

supported=$(nvcc --list-gpu-arch 2>/dev/null | sed -nE 's/^compute_([0-9]{2,3})$/\1/p' | sort -n -u)

gpu_lines=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null)
gpu_index=0
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    first=${CUDA_VISIBLE_DEVICES%%,*}
    first=$(printf '%s' "$first" | awk '{$1=$1; print}')
    if printf '%s\n' "$first" | grep -Eq '^[0-9]+$'; then
        gpu_count=$(printf '%s\n' "$gpu_lines" | awk 'NF { count++ } END { print count + 0 }')
        if [ "$first" -ge 0 ] && [ "$first" -lt "$gpu_count" ]; then
            gpu_index=$first
        fi
    fi
fi

gpu_line=$(printf '%s\n' "$gpu_lines" | awk -v n=$((gpu_index + 1)) 'NR == n { print; exit }')
detected=$(printf '%s\n' "$gpu_line" | sed -nE 's/.*,[[:space:]]*([0-9]+)[.]([0-9]+).*/\1\2/p')

if [ -n "$detected" ] && [ "$detected" -gt 0 ] 2>/dev/null; then
    if [ -z "$supported" ] || contains_number "$detected"; then
        printf 'compute_%s\n' "$detected"
        exit 0
    fi

    best=
    for arch in $supported; do
        if [ "$arch" -le "$detected" ]; then
            best=$arch
        fi
    done

    if [ -n "$best" ]; then
        printf 'compute_%s\n' "$best"
        exit 0
    fi
fi

fallback_number=$(arch_number "$fallback")
if [ -n "$supported" ] && [ "$fallback_number" -gt 0 ] 2>/dev/null && ! contains_number "$fallback_number"; then
    best_fallback=
    for arch in $supported; do
        best_fallback=$arch
    done

    if [ -n "$best_fallback" ]; then
        printf 'compute_%s\n' "$best_fallback"
        exit 0
    fi
fi

printf '%s\n' "$fallback"
