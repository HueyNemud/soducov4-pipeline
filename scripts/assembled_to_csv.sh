#!/usr/bin/env bash
set -euo pipefail

[[ $# -ne 1 ]] && { echo "Usage: $0 <fichier.json>" >&2; exit 1; }

# En-tête CSV
echo '"id","cat","alignment","text","name","activity","label","number","complement","raw_text","pages","lines"'

jq -r '
  .items | to_entries[] 
  | .key as $id 
  | .value as $root
  | (
      if $root.cat == "ent" and ($root.addresses | type == "array" and length > 0) 
      then $root.addresses[] 
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
      $root.raw_text,
      (
        $root.lines_resolved // [] 
        | if type == "array" then map(.page) | unique | join(";") else "" end
      ),
      (
        $root.lines_resolved // [] 
        | if type == "array" then map(.line) | unique | join(";") else "" end
      )
    ]
  | @csv
' "$1"