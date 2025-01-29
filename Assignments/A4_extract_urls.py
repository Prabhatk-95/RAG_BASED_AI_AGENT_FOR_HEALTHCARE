import re

paragraph = """
Look for data to help you address the question. Governments are good
sources because data from public research is often freely available. Good
places to start include http://www.data.gov/, and http://www.science.
gov/, and in the United Kingdom, http://data.gov.uk/.
Two of my favorite data sets are the General Social Survey at http://www3.norc.org/gss+website/,
and the European Social Survey at http://www.europeansocialsurvey.org/.
'''

"""

# Using regex to extract URLs
url_pattern = r"https?://[^\s,]+"  # Matches http/https URLs
urls = re.findall(url_pattern, paragraph)

# Printing results
print("Extracted URLs:")
for url in urls:
    print(url)
