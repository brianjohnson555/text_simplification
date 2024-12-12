"""Takes the input url, retreives source html code, replaces words within
source, generates new html file with replaced words."""

import requests
from readability import Document

response = requests.get('http://example.com')
doc = Document(response.content)
doc.title()
doc.summary()