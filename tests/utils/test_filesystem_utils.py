# -*- coding: utf-8 -*-
"""
Created on Fri May  7 22:11:02 2021

@author: freeridingeo
"""

import unittest
import logging
import os
import tempfile
from pathlib import Path

import fs
from fs.osfs import OSFS
from fs.errors import CreateFailed

import sys
sys.path.append("D:/Code/eotopia/utils")
import filesystem_utils as fu

logging.basicConfig(level=logging.DEBUG)

class TestFilesystemUtils(unittest.TestCase):

    def test_get_local_filesystem(self):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            filesystem = fu.get_filesystem(tmp_dir_name)
            self.assertTrue(isinstance(filesystem, OSFS))

            subfolder_path = os.path.join(tmp_dir_name, 'subfolder')

            with self.assertRaises(CreateFailed):
                fu.get_filesystem(subfolder_path, create=False)

            filesystem = fu.get_filesystem(subfolder_path, create=True)
            self.assertTrue(isinstance(filesystem, OSFS))

    def test_pathlib_support(self):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            path = Path(tmp_dir_name)
            filesystem = fu.get_filesystem(path)
            self.assertTrue(isinstance(filesystem, OSFS))

if __name__ == '__main__':
    unittest.main()