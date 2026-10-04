import unittest
from types import SimpleNamespace
import numpy as np
from route_b.policy import select_top4

class ProtectedSelection(unittest.TestCase):
    def test_masked_protected_logits_do_not_cancel_legal_proposal(self):
        state=SimpleNamespace(protected_slots=(0,),configuration_id='frozen')
        basis=SimpleNamespace(Q=np.eye(3,dtype=complex),basis_gauge_sha256='gauge')
        coefficients=np.zeros((2,2,25),complex)
        coefficients[1,0,1]=1;coefficients[1,1,2]=1
        logits=np.array([[-np.inf,-np.inf],[2.,1.]])
        selected=select_top4(logits,0.,coefficients,state,basis)
        self.assertEqual([a.drop_indices for a in selected],[(1,),(1,)])
        self.assertEqual(select_top4(logits,2.,coefficients,state,basis),[])

if __name__=='__main__':unittest.main()
