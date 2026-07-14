import re

html_scraped_data = "<p>Welcome to our site!</p> Click <a href='/link'>here</a> to download the document.<br><b>Thank you!</b>"

# Find anything starting with < and ending with > and delete it 
cleaned_html =re.sub(r'<.*?>', '', html_scraped_data)

print("Before:", html_scraped_data) 
print("After :", cleaned_html)