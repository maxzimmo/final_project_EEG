import streamlit as st

with open('index.html', 'r') as html_file:
    html_code = html_file.read()

# Render HTML
st.markdown(html_code, unsafe_allow_html=True)

# Import CSS
css_code = '''
<link rel="stylesheet" type="text/css" href="style.css">
'''
# Render CSS
st.markdown(css_code, unsafe_allow_html=True)

with open('script.js', 'r') as js_file:
    js_code = js_file.read()

# Render JavaScript
st.markdown(js_code, unsafe_allow_html=True)
