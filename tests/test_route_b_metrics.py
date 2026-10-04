import unittest

class CoverageTests(unittest.TestCase):
    def test_ratio_of_sums_per_parent_object(self):
        from route_b.preflight import cov
        rows=[{'object_id':'a','g':1.,'union_reference':1.},{'object_id':'a','g':0.,'union_reference':9.},{'object_id':'b','g':2.,'union_reference':2.}]
        self.assertAlmostEqual(cov(rows,'g'),.55)
        self.assertIsNone(cov([{'object_id':'a','g':0.,'union_reference':0.}],'g'))

if __name__=='__main__':unittest.main()
