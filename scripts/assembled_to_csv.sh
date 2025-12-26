#!/usr/bin/env bash
set -euo pipefail

[[ $# -ne 1 ]] && { echo "Usage: $0 <fichier.json>" >&2; exit 1; }

# En-tête CSV
echo '"id","cat","alignment","text","name","activity","label","number","complement","raw_text"'

jq -r '
  [path(..)|select(length==2)] as $paths 
  | to_entries[] 
  | .key as $id 
  | .value.items[] 
  | . as $root
  | (
      if .cat == "ent" and (.addresses | type == "array" and length > 0) 
      then .addresses[] 
      else {label: null, number: null, complement: null} 
      end
    ) as $addr
  | [
      $id,
      $root.cat,
      $root.alignment,
      $root.text,
      $root.name,
      $root.activity,
      $addr.label,
      $addr.number,
      $addr.complement,
      $root.raw_text
    ]
  | @csv
' "$1"