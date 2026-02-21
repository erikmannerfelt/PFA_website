
# Build all cached items
build-cache:
    ipython -c 'from format_radargrams import *; print(len(parse_all_radargrams(progress=True)))'

# Rebuild all cached items
rebuild-cache:
    if [[ -d cache/radargrams ]]; then rm -r cache/radargrams; fi
    if [[ -d web/static/radargrams ]]; then rm -r web/static/radargrams; fi
    ipython -c 'from format_radargrams import *; print(len(parse_all_radargrams(progress=True, redo_cache=True)))'

# Run the webserver
web:
    python webserver.py
    
# Run the webserver in debug mode
web-debug: 
    python -c 'from webserver import *; main(debug=True)'
