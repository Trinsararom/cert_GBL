import pytesseract
from PIL import Image
import cv2
from datetime import datetime
import os
import re
import zipfile
import io
import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Cert",
    layout = 'wide',
)

st.title('Cert Scraper')

# Initialize the Tesseract OCR
def initialize_tesseract():
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Initialize Tesseract
initialize_tesseract()
