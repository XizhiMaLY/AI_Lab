git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch langchain/L2-Memory.ipynb' \
--prune-empty --tag-name-filter cat -- --all
