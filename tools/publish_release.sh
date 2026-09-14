#!/bin/sh
# Run only after authenticating GitHub CLI for the intended repository.
set -eu
repo=${1:?Usage: publish_release.sh OWNER/REPOSITORY RESULTS_DIRECTORY [TAG]}
results=${2:?Results directory required}
tag=${3:-reconstructions-v1}
command -v gh >/dev/null 2>&1 || { echo 'Install GitHub CLI and run gh auth login first.' >&2; exit 1; }
gh repo view "$repo" >/dev/null
(cd "$results" && sha256sum -c SHA256SUMS)
if ! gh release view "$tag" --repo "$repo" >/dev/null 2>&1; then
 gh release create "$tag" --repo "$repo" --draft --title 'Strict-25 reconstruction parameters' --notes-file "$results/README.md"
fi
for asset in "$results"/*.zip "$results/manifest.json" "$results/SHA256SUMS" "$results/metrics.json"; do
 gh release upload "$tag" "$asset" --repo "$repo"
done
echo 'All assets uploaded. Verify downloads, then publish the draft release.'
