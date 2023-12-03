#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 23:20:47 2023

@author: soneitaraherimalala
"""


import unittest
from unittest.mock import patch
from io import StringIO
import sys
import os
import streamlit as st
import unittest
from unittest.mock import patch
from io import StringIO
import P7_script  # Importez le module P7_script correctement

class TestMyCode(unittest.TestCase):
    @patch('builtins.input', side_effect=["123"])
    def test_selected_client(self, mock_input):
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            P7_script.main()  # Assurez-vous que vous appelez la fonction appropriée de P7_script

        output = mock_stdout.getvalue()
        self.assertTrue("Score")  

    def test_nonexistent_client(self):
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            P7_script.main()  # Appelez la fonction appropriée de P7_script
       
        output = mock_stdout.getvalue()
        self.assertTrue("Le numéro du client n'existe pas dans le DataFrame.") #in output)

if __name__ == '__main__':
    unittest.main()