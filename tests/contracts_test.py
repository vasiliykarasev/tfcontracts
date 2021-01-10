"""Unit tests for basic, common contracts."""
try:
  from __init__ import *
except:
  pass

import os
os.environ['CUDA_VISIBLE_DEVICES'] =''

import tfcontracts
import unittest
import tensorflow as tf


class TypeCheckingContractTest(unittest.TestCase):
  def test_without_type_annotations(self):
    # Contract violations are acceptable here: function doesn't have type
    # annotations, and therefore cannot be type-checked.
    @tfcontracts.TypeCheckingContract()
    def add_two_objects_no_type_annotations(x, y):
      return x + y

    add_two_objects_no_type_annotations(0, 0)
    add_two_objects_no_type_annotations(x=0, y=0.5)

  def test_with_type_annotations(self):
    # Contract violations are detected here: function has type annotations and
    # passed value type differs from the expected one.
    @tfcontracts.TypeCheckingContract()
    def add_two_ints(x: int, y: int) -> int:
      return x + y

    add_two_ints(0, 0)
    with self.assertRaises(tfcontracts.errors.InvalidArgumentError):
      add_two_ints(0, 0.5)
    with self.assertRaises(tfcontracts.errors.InvalidArgumentError):
      add_two_ints(x=0, y=0.5)


class DTypeContractTest(unittest.TestCase):
  def test_single_dtype_contract(self):
    """Verifies a contract where only a single dtype is allowed."""
    @tfcontracts.DTypeContract(value=tf.float32)
    def add_two_tensors(x, y):
      return x + y

    add_two_tensors(tf.zeros(1, tf.float32), tf.zeros(1, tf.float32))

    # This will raise an exception: contract prohibits passing tf.int32.
    with self.assertRaises(tfcontracts.errors.InvalidArgumentError):
      add_two_tensors(tf.zeros(1, tf.int32), tf.zeros(1, tf.float32))

  def test_anyof_dtype_contract(self):
    """Verifies a contract where multiple dtype's are allowed."""
    @tfcontracts.DTypeContract(value=[tf.float32, tf.float64])
    def test_func(x, y):
      return tf.zeros_like(x)

    test_func(tf.zeros(1, tf.float64), tf.zeros(1, tf.float32))

  def test_check_argument_dtypes(self):
    """Checks the underlying function in dtype contract verification."""
    check_argument_dtype_recursive = tfcontracts.dtype_contract.check_argument_dtype_recursive
    tensor_seq = [tf.zeros(1, tf.float32), tf.zeros(1, tf.int32)]
    tensor_dict = {'a': tf.zeros(1, tf.float32), 'b': tf.zeros(1, tf.int32)}

    self.assertTrue(
        check_argument_dtype_recursive(tensor_seq,
                                       desired_dtype=[tf.float32, tf.int32]))
    self.assertTrue(
        check_argument_dtype_recursive(tensor_dict,
                                       desired_dtype=[tf.float32, tf.int32]))

    # Should return false: input consists of int32 and float32, but only int32
    # is allowed.
    self.assertFalse(
        check_argument_dtype_recursive(tensor_seq, desired_dtype=tf.int32))
    self.assertFalse(
        check_argument_dtype_recursive(tensor_dict, desired_dtype=tf.int32))
    # Returns true on various non-tensor types.
    self.assertTrue(check_argument_dtype_recursive('hello', tf.int32))
    self.assertTrue(check_argument_dtype_recursive(924, tf.int32))


class CombinedContractTest(unittest.TestCase):
  def test_combined_contract(self):
    # A CombinedContract that has no contracts provided to the ctor is valid,
    # but does not do anything.
    @tfcontracts.CombinedContract([])
    def test_func(x, y):
      return x + y

    self.assertEqual(3, test_func(1, 2))

if __name__ == '__main__':
  unittest.main()
