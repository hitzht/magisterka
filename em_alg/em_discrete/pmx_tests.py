import unittest
from pmx import pmx


class MyTestCase(unittest.TestCase):
    def test_pmx1(self):
        p1 = [8, 4, 7, 3, 6, 2, 5, 1, 9, 0]
        p2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        res = pmx(p1, p2, 3, 7)

        self.assertEqual(p1, [8, 4, 7, 3, 6, 2, 5, 1, 9, 0])
        self.assertEqual(p2, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(res, [0, 7, 4, 3, 6, 2, 5, 1, 8, 9])

    def test_pmx2(self):
        p1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        p2 = [4, 5, 2, 1, 8, 7, 6, 9, 3]

        res = pmx(p1, p2, 3, 6)

        self.assertEqual(p1, [1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(p2, [4, 5, 2, 1, 8, 7, 6, 9, 3])
        self.assertEqual(res, [1, 8, 2, 4, 5, 6, 7, 9, 3])

    def test_pmx3(self):
        p1 = [1, 5, 2, 8, 7, 4, 3, 6]
        p2 = [4, 2, 5, 8, 1, 3, 6, 7]

        res = pmx(p1, p2, 2, 4)

        self.assertEqual(p1, [1, 5, 2, 8, 7, 4, 3, 6])
        self.assertEqual(p2, [4, 2, 5, 8, 1, 3, 6, 7])
        self.assertEqual(res, [4, 5, 2, 8, 7, 3, 6, 1])



if __name__ == '__main__':
    unittest.main()
