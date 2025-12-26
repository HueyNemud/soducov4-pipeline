#!/bin/bash

# Script to extract pages from PDF using pdftk
# Usage: ./extract_pages.sh csv_file
# Or: ./extract_pages.sh pdf_file start_page end_page output_file

if [ $# -eq 1 ] && [ -f "$1" ] && [[ "$1" == *.csv ]]; then
    # Process CSV file: assumes format pdf_file,start_page,end_page,output_file
    while IFS=, read -r pdf start end output; do
        if [ -n "$pdf" ] && [ -n "$start" ] && [ -n "$end" ] && [ -n "$output" ]; then
            pdftk "$pdf" cat "$start"-"$end" output "$output"
        fi
    done < "$1"
elif [ $# -eq 4 ]; then
    # Direct parameters
    pdf="$1"
    start="$2"
    end="$3"
    output="$4"
    pdftk "$pdf" cat "$start"-"$end" output "$output"
else
    echo "Usage: $0 csv_file (CSV format: pdf_file,start_page,end_page,output_file)"
    echo "Or: $0 pdf_file start_page end_page output_file"
    exit 1
fi