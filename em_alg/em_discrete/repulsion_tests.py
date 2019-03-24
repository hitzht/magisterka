import unittest
from repulsion import repulsion
from hamming import hamming_distance

class MyTestCase(unittest.TestCase):
    def test_repulsion(self):
        p1 = [1, 2, 3, 4, 5, 6, 7, 8]
        p2 = [1, 5, 2, 8, 7, 4, 3, 6]

        start_distance = hamming_distance(p1, p2)
        res = repulsion(p1, p2)
        self.assertEqual(p1, [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(p2, [1, 5, 2, 8, 7, 4, 3, 6])

        end_distance = hamming_distance(p1, res)
        self.assertTrue(end_distance >= start_distance)

if __name__ == '__main__':
    unittest.main()
