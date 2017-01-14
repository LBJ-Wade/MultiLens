
# Copyright (C) 2016 ETH Zurich, Institute for Astronomy

"""
Tests for `MultiLens` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import pytest
from MultiLens import MultiLens


class TestMultilens(object):

    def setup(self):
        #prepare unit test. Load data etc
        print("setting up " + __name__)
        pass

    def test_something(self):
        x = 1
        assert x==1

    def teardown(self):
        #tidy up
        print("tearing down " + __name__)
        pass

if __name__ == '__main__':
    pytest.main()