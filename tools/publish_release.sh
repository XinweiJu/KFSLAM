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
 gh release create "$tag" --repo "$repo" --draft --title 'Reconstruction Parameters' --notes-file "$results/README.md"
fi
python3 - "$results" "$tag" "$repo" <<'PYUPLOAD'
import json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);manifest=json.loads((root/'manifest.json').read_text())
assets=[x['asset'] for x in manifest]+['manifest.json','SHA256SUMS','metrics.json']
for name in assets:
 if Path(name).name!=name:raise ValueError('Invalid asset name')
 subprocess.run(['gh','release','upload',sys.argv[2],str(root/name),'--repo',sys.argv[3]],check=True)
PYUPLOAD
echo 'All assets uploaded. Verify downloads, then publish the draft release.'
