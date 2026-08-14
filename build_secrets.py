"""
One-off helper: builds .streamlit/secrets.toml from your downloaded Google
Cloud service-account JSON key, so the tricky private_key field never has
to be hand-copied (that's what usually breaks this file).

Run with:  python3 build_secrets.py
"""

import json
from pathlib import Path

json_path = input("Path to the downloaded service-account JSON file: ").strip().strip('"').strip("'")
sheet_url = input("Your Google Sheet's URL: ").strip().strip('"').strip("'")

with open(json_path) as f:
    creds = json.load(f)

required = [
    "type", "project_id", "private_key_id", "private_key", "client_email",
    "client_id", "auth_uri", "token_uri", "auth_provider_x509_cert_url",
    "client_x509_cert_url",
]
missing = [k for k in required if k not in creds]
if missing:
    raise SystemExit(f"That JSON file is missing field(s): {missing} - did you download the right key?")

lines = ["[connections.gsheets]", f"spreadsheet = {json.dumps(sheet_url)}"]
for key in required:
    lines.append(f"{key} = {json.dumps(creds[key])}")

out_path = Path(".streamlit/secrets.toml")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("\n".join(lines) + "\n")

print(f"\nWrote {out_path.resolve()}")
print("Run 'streamlit run WA.py' now.")
