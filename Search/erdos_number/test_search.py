from search import bidirectional_wiki_search

s = "Chess"
e = "Terence Tao"
print(f"Testing search between '{s}' and '{e}'...")
path = bidirectional_wiki_search(s, e, max_depth=3)
print(f"Result: {path}")
