"""Unit tests for basic, common contracts."""
try:
  from __init__ import *
except:
  pass

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tfcontracts
import unittest
import tensorflow as tf


class AssertUtilitiesTest(unittest.TestCase):
  def test_assert_shapes_same_empty(self):
    tfcontracts.assert_shapes_same([])

  def test_assert_shapes_same_equal_shapes(self):
    tfcontracts.assert_shapes_same([tf.zeros([1, 2, 3]), tf.zeros([1, 2, 3])])

  def test_assert_shapes_raises(self):
    with self.assertRaises(ValueError):
      tfcontracts.assert_shapes_same([tf.zeros([1, 2]), tf.zeros([2])])

  def test_assert_in_interval_scalar(self):
    tfcontracts.assert_in_interval(tf.constant([0.5], tf.float32), 0.0, 1.0)

  def test_assert_in_interval_tensor(self):
    tfcontracts.assert_in_interval(tf.constant([-1, -2, -3], tf.float32),
                                   low=tf.constant([-2, -3, -4], tf.float32),
                                   high=tf.zeros([3], tf.float32))

  def test_assert_in_interval_raises(self):
    with self.assertRaises(tf.errors.InvalidArgumentError):
      tfcontracts.assert_in_interval(tf.constant([0.0]), low=1.0, high=2.0)


if __name__ == '__main__':
  unittest.main()
