import re

# r = "(hi|hello|hey)[ ]*([a-z]*)"
# print(re.match(r, 'Hello Rosa', flags=re.IGNORECASE))
# print(re.match(r, "hi no, hi no, it's off to work ...", flags=re.IGNORECASE))
# print(re.match(r, "hey, what's up", flags=re.IGNORECASE))

r = r"[^a-z]*([y]o|[h']?ello|ok|hey|(good[ ])?(morn[gin']{0,3}|" \
    r"afternoon|even[gin']{0,3}))[\s,;:]{1,3}([a-z]{1,20})"
re_greeting = re.compile(r, flags=re.IGNORECASE)
print(re_greeting.match('Hello Rosa'))
print(re_greeting.match('Hello Rosa').groups())
print(re_greeting.match("Good morning Rosa"))
print(re_greeting.match("Good evening Rosa Parks").groups())





