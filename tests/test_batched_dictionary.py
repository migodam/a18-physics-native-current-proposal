import unittest
import numpy as np
from test_shared_backend import dense_backend
from shared_backend.discrete_audit import audit_bank
from shared_backend.batched_dictionary import audit_bank_batched

class BatchedDictionaryTests(unittest.TestCase):
    def test_matches_scalar_endpoints_including_duplicate_columns(self):
        b,_,_,_=dense_backend(n=30);s=b.freeze_state(b.get_receiver_basis(4),object_id='x',state_id='0');rng=np.random.default_rng(203)
        bank=np.column_stack((s.U,rng.normal(size=(30,8))+1j*rng.normal(size=(30,8))));J=b.full_J_action(np.eye(4))
        a=audit_bank(b,s,bank,J,J);z=audit_bank_batched(b,s,bank,J,J)
        np.testing.assert_array_equal(a['valid'],z['valid']);np.testing.assert_allclose(a['gains'],z['gains'],atol=1e-10,equal_nan=True);np.testing.assert_allclose(a['tau'],z['tau'],rtol=1e-10,equal_nan=True);np.testing.assert_allclose(a['scores'],z['scores'],atol=1e-10)

if __name__=='__main__':unittest.main()
