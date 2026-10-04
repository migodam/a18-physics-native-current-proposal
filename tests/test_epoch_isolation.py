import unittest
import numpy as np
from test_shared_backend import dense_backend
from shared_backend.isolation import cached_only_epoch

class IsolationTests(unittest.TestCase):
    def test_real_callbacks_fail_before_execution_in_cached_epoch(self):
        b,_,_,_=dense_backend();before=b.receipt.snapshot()
        for name,x in (('apply_L',np.zeros(12)),('apply_S',np.zeros(12)),('apply_B',np.zeros(4)),('full_J_action',np.zeros(4))):
            with self.assertRaises(RuntimeError):
                with cached_only_epoch():getattr(b,name)(x)
        self.assertEqual(before,b.receipt.snapshot())

if __name__=='__main__':unittest.main()
