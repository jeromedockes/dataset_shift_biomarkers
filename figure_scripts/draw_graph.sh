#! /bin/bash

tempdir=$(mktemp -d)
trap 'rm -rf $tempdir' EXIT
pdf_file="$tempdir/graph.pdf"
dot "$1" -Tpdf > "$pdf_file"
grep -q "CROP_PDF" "$1" && pdfcrop --margins "1 1 1 1" "$pdf_file" "$pdf_file"
cat "$pdf_file"
