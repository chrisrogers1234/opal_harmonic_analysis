import math
import numpy
import unittest
import transfer_map

class TestTransferMap(unittest.TestCase):
    def setUp(self):
        self.verbose = 1

    def _test_transfer_map(self):
        tm = transfer_map.TransferMap()
        tm.max_order = 2
        tm.dimension = 4
        tm.momentum = ((70+938.272)**2-938.272**2)**0.5
        print(tm.calculate_delta_map([0, 0, 1], [0, 2]))

    def _test_get_multipole_polynomial(self):
        tm = transfer_map.TransferMap()
        tm.max_order = 2
        tm.dimension = 4
        tm.get_indices()

        test_poly = tm.get_multipole_polynomial(0, 2+1j) # dipole
        ref_poly = numpy.array([2.0+1.0j]+[0.0 for i in tm.indices[1:]])
        if self.verbose:
            print("    ", tm.indices)
            print("Test", test_poly)
            print("Ref ", ref_poly)
        for item in zip(test_poly, ref_poly):
            self.assertEqual(item[0], item[1])

        test_poly = tm.get_multipole_polynomial(1, 3+4j) # quadrupole
        ref_poly = numpy.array([0.0, 3.0+4.0j, 0.0, 3.0j-4.0]+[0.0 for i in tm.indices])
        if self.verbose:
            print()
            print("    ", tm.indices)
            print("Test", test_poly)
            print("Ref ", ref_poly)
        for item in zip(test_poly, ref_poly):
            self.assertEqual(item[0], item[1])

        test_poly = tm.get_multipole_polynomial(2, 4+5j) # sextupole
        ref_poly = numpy.array([0.0]*5+[4+5j, 0.0, 2j*(4+5j)]+[0.0]*4+[1j**2*(4+5j)]+[0.0]*2)
        if self.verbose:
            print()
            print("    ", tm.indices)
            print("Test", test_poly)
            print("Ref ", ref_poly)
        for item in zip(test_poly, ref_poly):
            self.assertEqual(item[0], item[1])

        tm.max_order = 3
        tm.get_indices()
        test_poly = tm.get_multipole_polynomial(3, 1+2j) # octupole
        ref_poly = numpy.array([0.0]*15+[1+2j, 0.0, 3j*(1+2j)]+[0.0]*4+[3*(1j)**2*(1+2j)]+[0.0]*8+[(1j)**3*(1+2j)]+[0.0]*3)
        if self.verbose:
            print()
            print("    ", tm.indices)
            print("Test", test_poly)
            print("Ref ", ref_poly)
        for item in zip(test_poly, ref_poly):
            self.assertEqual(item[0], item[1])

    def _test_get_multipole_field(self):
        tm = transfer_map.TransferMap()
        tm.max_order = 2
        tm.dimension = 4
        tm.momentum = ((70+938.272)**2-938.272**2)**0.5
        tm.get_indices()
        cn = [0.0, 1+2j, 0.0, 0.0, 0.0]
        bx_test, by_test = tm.get_multipole_field(cn)
        if self.verbose:
            print(tm.indices)
            print("Bx", bx_test)
            print("By", by_test)
        bx_ref = [0.0, 2.0, 0.0, 1.0]+[0.0]*(len(tm.indices)-4)
        by_ref = [0.0, 1.0, 0.0, -2.0]+[0.0]*(len(tm.indices)-4)
        for ref, test in zip(bx_ref, bx_test):
            self.assertEqual(ref, test)
        for ref, test in zip(by_ref, by_test):
            self.assertEqual(ref, test)


    def test_calculate_delta_map_rotated(self):
        tm = transfer_map.TransferMap()
        tm.max_order = 2
        tm.dimension = 4
        tm.momentum = ((70+938.272)**2-938.272**2)**0.5
        tm.get_indices()
        cn = [0.0, 3+0j, 0.0, 0.0]
        test_tm = tm.calculate_delta_map_rotated(cn, math.pi/4)

        pinv = 1/tm.momentum
        b_units = tm.c_light*1e-9/tm.momentum

        npoints = len(tm.get_indices())
        ref_tm = numpy.array([
            [0.0]*2+[pinv]+[0.0]*(npoints-3),
            [0.0]*npoints,
            [0.0]*4+[pinv]+[0.0]*(npoints-5),
            [0.0]*npoints,
        ])
        if self.verbose:
            print()
            print(tm.indices)
            bx, by = tm.get_multipole_field(cn)
            print("Field Bx")
            print(bx)
            print("Field By")
            print(by)
            print("Test")
            print(test_tm)
            print("Ref ")
            print(ref_tm)
        for test_row, ref_row in zip(test_tm, ref_tm):
            for test_cell, ref_cell in zip(test_row, ref_row):
                pass #self.assertEqual(test_cell, ref_cell)
        return
        tm.max_order = 3
        tm.get_indices()
        cn = [0.0, 0.0, 2.0+0.0j, 0.0]
        test_tm = tm.calculate_delta_map_rotated(cn, 0.0)
        if self.verbose:
            print()
            print(tm.indices)
            bx, by = tm.get_multipole_field(cn)
            print("Field Bx")
            print(bx)
            print("Field By")
            print(by)
            print("Test")
            print(test_tm)
            #print("Ref ")
            #print(ref_tm)
        test_tm = tm.calculate_delta_map_rotated(cn, math.pi/4)
        if self.verbose:
            print()
            print(tm.indices)
            bx, by = tm.get_multipole_field(cn)
            print("Field Bx")
            print(bx)
            print("Field By")
            print(by)
            print("Test")
            print(test_tm)
            #print("Ref ")
            #print(ref_tm)



if __name__ == "__main__":
    numpy.set_printoptions(linewidth=200)
    unittest.main()