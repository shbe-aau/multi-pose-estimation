# Remove trailing whitespace, and replace tabs with the appropriate number of spaces
for file in `find . -name "*.py"`; do
    echo "$file"; emacs --batch "$file" \
        --eval '(delete-trailing-whitespace)' \
        -f 'save-buffer'
done
