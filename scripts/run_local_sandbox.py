"""macOS sandbox: protect home files, restrict writes, allow loopback networking."""
from pathlib import Path
import json
import os
import sys

project = Path(__file__).resolve().parents[1]
profile = project / "runs" / "local_server.sb"
profile.parent.mkdir(parents=True, exist_ok=True)
rules = ["(version 1)", "(allow default)"]
# Runtime/system reads remain available; home file contents outside this project
# are blocked. Metadata reads are needed to traverse parent directories.
rules.append(f"(deny file-read-data (require-all (subpath {json.dumps(str(Path.home()))}) "
             f"(require-not (subpath {json.dumps(str(project))}))))")
outside_writes = " ".join(f"(require-not (subpath {json.dumps(str(project / name))}))" for name in ("runs", "configs"))
rules.append(f'(deny file-write* (require-all {outside_writes} (require-not (literal "/dev/null"))))')
rules += ['(deny network-inbound (require-not (local ip "localhost:*")))',
          '(deny network-outbound (require-not (remote ip "localhost:*")))',
          '(deny network-bind (require-not (local ip "localhost:*")))']
profile.write_text("\n".join(rules) + "\n", encoding="utf-8")
os.chdir(project)
os.execv("/usr/bin/sandbox-exec", ["sandbox-exec", "-f", str(profile), sys.executable, "-B", "-m",
                                 "eeg_speller.web.server", *sys.argv[1:]])
