# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
# pylint: skip-file
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: task.proto

from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='task.proto',
  package='automl_zero',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=b'\n\ntask.proto\x12\x0b\x61utoml_zero\"6\n\x0eTaskCollection\x12$\n\x05tasks\x18\x01 \x03(\x0b\x32\x15.automl_zero.TaskSpec\"\xa7\x06\n\x08TaskSpec\x12\x15\n\rfeatures_size\x18\r \x01(\x05\x12\x1a\n\x12num_train_examples\x18\x01 \x01(\x05\x12\x1b\n\x10num_train_epochs\x18\x15 \x01(\x05:\x01\x31\x12\x1a\n\x12num_valid_examples\x18\x02 \x01(\x05\x12\x11\n\tnum_tasks\x18\x03 \x01(\x05\x12\x12\n\ndata_seeds\x18\x04 \x03(\r\x12\x13\n\x0bparam_seeds\x18\x05 \x03(\r\x12(\n\teval_type\x18\x1c \x01(\x0e\x32\x15.automl_zero.EvalType\x12T\n\x1dscalar_linear_regression_task\x18\x06 \x01(\x0b\x32+.automl_zero.ScalarLinearRegressionTaskSpecH\x00\x12Y\n scalar_2layer_nn_regression_task\x18\x07 \x01(\x0b\x32-.automl_zero.Scalar2LayerNNRegressionTaskSpecH\x00\x12^\n$projected_binary_classification_task\x18\x18 \x01(\x0b\x32..automl_zero.ProjectedBinaryClassificationTaskH\x00\x12>\n\x14unit_test_fixed_task\x18( \x01(\x0b\x32\x1e.automl_zero.UnitTestFixedTaskH\x00\x12\x42\n\x14unit_test_zeros_task\x18- \x01(\x0b\x32\".automl_zero.UnitTestZerosTaskSpecH\x00\x12@\n\x13unit_test_ones_task\x18. \x01(\x0b\x32!.automl_zero.UnitTestOnesTaskSpecH\x00\x12J\n\x18unit_test_increment_task\x18/ \x01(\x0b\x32&.automl_zero.UnitTestIncrementTaskSpecH\x00\x12\x19\n\x11num_test_examples\x18\x12 \x01(\x05\x42\x0b\n\ttask_type\" \n\x1eScalarLinearRegressionTaskSpec\"\"\n Scalar2LayerNNRegressionTaskSpec\"\xb5\x02\n!ProjectedBinaryClassificationTask\x12\x16\n\x0epositive_class\x18\x01 \x01(\x05\x12\x16\n\x0enegative_class\x18\x02 \x01(\x05\x12\x14\n\x0c\x64\x61taset_name\x18\x03 \x01(\t\x12\x0e\n\x04path\x18\x04 \x01(\tH\x00\x12\x32\n\x07\x64\x61taset\x18\x05 \x01(\x0b\x32\x1f.automl_zero.ScalarLabelDatasetH\x00\x12.\n\x0eheld_out_pairs\x18\x06 \x03(\x0b\x32\x16.automl_zero.ClassPair\x12\"\n\x17min_supported_data_seed\x18\x07 \x01(\x05:\x01\x30\x12#\n\x17max_supported_data_seed\x18\x08 \x01(\x05:\x02\x31\x30\x42\r\n\x0btask_source\";\n\tClassPair\x12\x16\n\x0epositive_class\x18\x01 \x01(\x05\x12\x16\n\x0enegative_class\x18\x02 \x01(\x05\"\xf0\x01\n\x12ScalarLabelDataset\x12\x32\n\x0etrain_features\x18\x01 \x03(\x0b\x32\x1a.automl_zero.FeatureVector\x12\x14\n\x0ctrain_labels\x18\x02 \x03(\x02\x12\x32\n\x0evalid_features\x18\x03 \x03(\x0b\x32\x1a.automl_zero.FeatureVector\x12\x14\n\x0cvalid_labels\x18\x04 \x03(\x02\x12\x31\n\rtest_features\x18\x05 \x03(\x0b\x32\x1a.automl_zero.FeatureVector\x12\x13\n\x0btest_labels\x18\x06 \x03(\x02\"!\n\rFeatureVector\x12\x10\n\x08\x66\x65\x61tures\x18\x01 \x03(\x02\"\x85\x03\n\x11UnitTestFixedTask\x12<\n\x0etrain_features\x18\x01 \x03(\x0b\x32$.automl_zero.UnitTestFixedTaskVector\x12:\n\x0ctrain_labels\x18\x02 \x03(\x0b\x32$.automl_zero.UnitTestFixedTaskVector\x12<\n\x0evalid_features\x18\x03 \x03(\x0b\x32$.automl_zero.UnitTestFixedTaskVector\x12:\n\x0cvalid_labels\x18\x04 \x03(\x0b\x32$.automl_zero.UnitTestFixedTaskVector\x12;\n\rtest_features\x18\x05 \x03(\x0b\x32$.automl_zero.UnitTestFixedTaskVector\x12\x39\n\x0btest_labels\x18\x06 \x03(\x0b\x32$.automl_zero.UnitTestFixedTaskVectorJ\x04\x08\x07\x10\x08\"+\n\x17UnitTestFixedTaskVector\x12\x10\n\x08\x65lements\x18\x01 \x03(\x01\"\x17\n\x15UnitTestZerosTaskSpec\"\x16\n\x14UnitTestOnesTaskSpec\"1\n\x19UnitTestIncrementTaskSpec\x12\x14\n\tincrement\x18\x01 \x01(\x01:\x01\x31*>\n\x08\x45valType\x12\x15\n\x11INVALID_EVAL_TYPE\x10\x00\x12\r\n\tRMS_ERROR\x10\x01\x12\x0c\n\x08\x41\x43\x43URACY\x10\x04*$\n\x0e\x41\x63tivationType\x12\x08\n\x04RELU\x10\x00\x12\x08\n\x04TANH\x10\x01'
)

_EVALTYPE = _descriptor.EnumDescriptor(
  name='EvalType',
  full_name='automl_zero.EvalType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='INVALID_EVAL_TYPE', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RMS_ERROR', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ACCURACY', index=2, number=4,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=2151,
  serialized_end=2213,
)
_sym_db.RegisterEnumDescriptor(_EVALTYPE)

EvalType = enum_type_wrapper.EnumTypeWrapper(_EVALTYPE)
_ACTIVATIONTYPE = _descriptor.EnumDescriptor(
  name='ActivationType',
  full_name='automl_zero.ActivationType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='RELU', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TANH', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=2215,
  serialized_end=2251,
)
_sym_db.RegisterEnumDescriptor(_ACTIVATIONTYPE)

ActivationType = enum_type_wrapper.EnumTypeWrapper(_ACTIVATIONTYPE)
INVALID_EVAL_TYPE = 0
RMS_ERROR = 1
ACCURACY = 4
RELU = 0
TANH = 1



_TASKCOLLECTION = _descriptor.Descriptor(
  name='TaskCollection',
  full_name='automl_zero.TaskCollection',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='tasks', full_name='automl_zero.TaskCollection.tasks', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=27,
  serialized_end=81,
)


_TASKSPEC = _descriptor.Descriptor(
  name='TaskSpec',
  full_name='automl_zero.TaskSpec',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='features_size', full_name='automl_zero.TaskSpec.features_size', index=0,
      number=13, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_train_examples', full_name='automl_zero.TaskSpec.num_train_examples', index=1,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_train_epochs', full_name='automl_zero.TaskSpec.num_train_epochs', index=2,
      number=21, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_valid_examples', full_name='automl_zero.TaskSpec.num_valid_examples', index=3,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_tasks', full_name='automl_zero.TaskSpec.num_tasks', index=4,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_seeds', full_name='automl_zero.TaskSpec.data_seeds', index=5,
      number=4, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='param_seeds', full_name='automl_zero.TaskSpec.param_seeds', index=6,
      number=5, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='eval_type', full_name='automl_zero.TaskSpec.eval_type', index=7,
      number=28, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scalar_linear_regression_task', full_name='automl_zero.TaskSpec.scalar_linear_regression_task', index=8,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scalar_2layer_nn_regression_task', full_name='automl_zero.TaskSpec.scalar_2layer_nn_regression_task', index=9,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='projected_binary_classification_task', full_name='automl_zero.TaskSpec.projected_binary_classification_task', index=10,
      number=24, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='unit_test_fixed_task', full_name='automl_zero.TaskSpec.unit_test_fixed_task', index=11,
      number=40, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='unit_test_zeros_task', full_name='automl_zero.TaskSpec.unit_test_zeros_task', index=12,
      number=45, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='unit_test_ones_task', full_name='automl_zero.TaskSpec.unit_test_ones_task', index=13,
      number=46, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='unit_test_increment_task', full_name='automl_zero.TaskSpec.unit_test_increment_task', index=14,
      number=47, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_test_examples', full_name='automl_zero.TaskSpec.num_test_examples', index=15,
      number=18, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='task_type', full_name='automl_zero.TaskSpec.task_type',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=84,
  serialized_end=891,
)


_SCALARLINEARREGRESSIONTASKSPEC = _descriptor.Descriptor(
  name='ScalarLinearRegressionTaskSpec',
  full_name='automl_zero.ScalarLinearRegressionTaskSpec',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=893,
  serialized_end=925,
)


_SCALAR2LAYERNNREGRESSIONTASKSPEC = _descriptor.Descriptor(
  name='Scalar2LayerNNRegressionTaskSpec',
  full_name='automl_zero.Scalar2LayerNNRegressionTaskSpec',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=927,
  serialized_end=961,
)


_PROJECTEDBINARYCLASSIFICATIONTASK = _descriptor.Descriptor(
  name='ProjectedBinaryClassificationTask',
  full_name='automl_zero.ProjectedBinaryClassificationTask',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='positive_class', full_name='automl_zero.ProjectedBinaryClassificationTask.positive_class', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='negative_class', full_name='automl_zero.ProjectedBinaryClassificationTask.negative_class', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dataset_name', full_name='automl_zero.ProjectedBinaryClassificationTask.dataset_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='path', full_name='automl_zero.ProjectedBinaryClassificationTask.path', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dataset', full_name='automl_zero.ProjectedBinaryClassificationTask.dataset', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='held_out_pairs', full_name='automl_zero.ProjectedBinaryClassificationTask.held_out_pairs', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='min_supported_data_seed', full_name='automl_zero.ProjectedBinaryClassificationTask.min_supported_data_seed', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_supported_data_seed', full_name='automl_zero.ProjectedBinaryClassificationTask.max_supported_data_seed', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=10,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='task_source', full_name='automl_zero.ProjectedBinaryClassificationTask.task_source',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=964,
  serialized_end=1273,
)


_CLASSPAIR = _descriptor.Descriptor(
  name='ClassPair',
  full_name='automl_zero.ClassPair',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='positive_class', full_name='automl_zero.ClassPair.positive_class', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='negative_class', full_name='automl_zero.ClassPair.negative_class', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1275,
  serialized_end=1334,
)


_SCALARLABELDATASET = _descriptor.Descriptor(
  name='ScalarLabelDataset',
  full_name='automl_zero.ScalarLabelDataset',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='train_features', full_name='automl_zero.ScalarLabelDataset.train_features', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='train_labels', full_name='automl_zero.ScalarLabelDataset.train_labels', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='valid_features', full_name='automl_zero.ScalarLabelDataset.valid_features', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='valid_labels', full_name='automl_zero.ScalarLabelDataset.valid_labels', index=3,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='test_features', full_name='automl_zero.ScalarLabelDataset.test_features', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='test_labels', full_name='automl_zero.ScalarLabelDataset.test_labels', index=5,
      number=6, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1337,
  serialized_end=1577,
)


_FEATUREVECTOR = _descriptor.Descriptor(
  name='FeatureVector',
  full_name='automl_zero.FeatureVector',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='features', full_name='automl_zero.FeatureVector.features', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1579,
  serialized_end=1612,
)


_UNITTESTFIXEDTASK = _descriptor.Descriptor(
  name='UnitTestFixedTask',
  full_name='automl_zero.UnitTestFixedTask',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='train_features', full_name='automl_zero.UnitTestFixedTask.train_features', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='train_labels', full_name='automl_zero.UnitTestFixedTask.train_labels', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='valid_features', full_name='automl_zero.UnitTestFixedTask.valid_features', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='valid_labels', full_name='automl_zero.UnitTestFixedTask.valid_labels', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='test_features', full_name='automl_zero.UnitTestFixedTask.test_features', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='test_labels', full_name='automl_zero.UnitTestFixedTask.test_labels', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1615,
  serialized_end=2004,
)


_UNITTESTFIXEDTASKVECTOR = _descriptor.Descriptor(
  name='UnitTestFixedTaskVector',
  full_name='automl_zero.UnitTestFixedTaskVector',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='elements', full_name='automl_zero.UnitTestFixedTaskVector.elements', index=0,
      number=1, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=2006,
  serialized_end=2049,
)


_UNITTESTZEROSTASKSPEC = _descriptor.Descriptor(
  name='UnitTestZerosTaskSpec',
  full_name='automl_zero.UnitTestZerosTaskSpec',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=2051,
  serialized_end=2074,
)


_UNITTESTONESTASKSPEC = _descriptor.Descriptor(
  name='UnitTestOnesTaskSpec',
  full_name='automl_zero.UnitTestOnesTaskSpec',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=2076,
  serialized_end=2098,
)


_UNITTESTINCREMENTTASKSPEC = _descriptor.Descriptor(
  name='UnitTestIncrementTaskSpec',
  full_name='automl_zero.UnitTestIncrementTaskSpec',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='increment', full_name='automl_zero.UnitTestIncrementTaskSpec.increment', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=2100,
  serialized_end=2149,
)

_TASKCOLLECTION.fields_by_name['tasks'].message_type = _TASKSPEC
_TASKSPEC.fields_by_name['eval_type'].enum_type = _EVALTYPE
_TASKSPEC.fields_by_name['scalar_linear_regression_task'].message_type = _SCALARLINEARREGRESSIONTASKSPEC
_TASKSPEC.fields_by_name['scalar_2layer_nn_regression_task'].message_type = _SCALAR2LAYERNNREGRESSIONTASKSPEC
_TASKSPEC.fields_by_name['projected_binary_classification_task'].message_type = _PROJECTEDBINARYCLASSIFICATIONTASK
_TASKSPEC.fields_by_name['unit_test_fixed_task'].message_type = _UNITTESTFIXEDTASK
_TASKSPEC.fields_by_name['unit_test_zeros_task'].message_type = _UNITTESTZEROSTASKSPEC
_TASKSPEC.fields_by_name['unit_test_ones_task'].message_type = _UNITTESTONESTASKSPEC
_TASKSPEC.fields_by_name['unit_test_increment_task'].message_type = _UNITTESTINCREMENTTASKSPEC
_TASKSPEC.oneofs_by_name['task_type'].fields.append(
  _TASKSPEC.fields_by_name['scalar_linear_regression_task'])
_TASKSPEC.fields_by_name['scalar_linear_regression_task'].containing_oneof = _TASKSPEC.oneofs_by_name['task_type']
_TASKSPEC.oneofs_by_name['task_type'].fields.append(
  _TASKSPEC.fields_by_name['scalar_2layer_nn_regression_task'])
_TASKSPEC.fields_by_name['scalar_2layer_nn_regression_task'].containing_oneof = _TASKSPEC.oneofs_by_name['task_type']
_TASKSPEC.oneofs_by_name['task_type'].fields.append(
  _TASKSPEC.fields_by_name['projected_binary_classification_task'])
_TASKSPEC.fields_by_name['projected_binary_classification_task'].containing_oneof = _TASKSPEC.oneofs_by_name['task_type']
_TASKSPEC.oneofs_by_name['task_type'].fields.append(
  _TASKSPEC.fields_by_name['unit_test_fixed_task'])
_TASKSPEC.fields_by_name['unit_test_fixed_task'].containing_oneof = _TASKSPEC.oneofs_by_name['task_type']
_TASKSPEC.oneofs_by_name['task_type'].fields.append(
  _TASKSPEC.fields_by_name['unit_test_zeros_task'])
_TASKSPEC.fields_by_name['unit_test_zeros_task'].containing_oneof = _TASKSPEC.oneofs_by_name['task_type']
_TASKSPEC.oneofs_by_name['task_type'].fields.append(
  _TASKSPEC.fields_by_name['unit_test_ones_task'])
_TASKSPEC.fields_by_name['unit_test_ones_task'].containing_oneof = _TASKSPEC.oneofs_by_name['task_type']
_TASKSPEC.oneofs_by_name['task_type'].fields.append(
  _TASKSPEC.fields_by_name['unit_test_increment_task'])
_TASKSPEC.fields_by_name['unit_test_increment_task'].containing_oneof = _TASKSPEC.oneofs_by_name['task_type']
_PROJECTEDBINARYCLASSIFICATIONTASK.fields_by_name['dataset'].message_type = _SCALARLABELDATASET
_PROJECTEDBINARYCLASSIFICATIONTASK.fields_by_name['held_out_pairs'].message_type = _CLASSPAIR
_PROJECTEDBINARYCLASSIFICATIONTASK.oneofs_by_name['task_source'].fields.append(
  _PROJECTEDBINARYCLASSIFICATIONTASK.fields_by_name['path'])
_PROJECTEDBINARYCLASSIFICATIONTASK.fields_by_name['path'].containing_oneof = _PROJECTEDBINARYCLASSIFICATIONTASK.oneofs_by_name['task_source']
_PROJECTEDBINARYCLASSIFICATIONTASK.oneofs_by_name['task_source'].fields.append(
  _PROJECTEDBINARYCLASSIFICATIONTASK.fields_by_name['dataset'])
_PROJECTEDBINARYCLASSIFICATIONTASK.fields_by_name['dataset'].containing_oneof = _PROJECTEDBINARYCLASSIFICATIONTASK.oneofs_by_name['task_source']
_SCALARLABELDATASET.fields_by_name['train_features'].message_type = _FEATUREVECTOR
_SCALARLABELDATASET.fields_by_name['valid_features'].message_type = _FEATUREVECTOR
_SCALARLABELDATASET.fields_by_name['test_features'].message_type = _FEATUREVECTOR
_UNITTESTFIXEDTASK.fields_by_name['train_features'].message_type = _UNITTESTFIXEDTASKVECTOR
_UNITTESTFIXEDTASK.fields_by_name['train_labels'].message_type = _UNITTESTFIXEDTASKVECTOR
_UNITTESTFIXEDTASK.fields_by_name['valid_features'].message_type = _UNITTESTFIXEDTASKVECTOR
_UNITTESTFIXEDTASK.fields_by_name['valid_labels'].message_type = _UNITTESTFIXEDTASKVECTOR
_UNITTESTFIXEDTASK.fields_by_name['test_features'].message_type = _UNITTESTFIXEDTASKVECTOR
_UNITTESTFIXEDTASK.fields_by_name['test_labels'].message_type = _UNITTESTFIXEDTASKVECTOR
DESCRIPTOR.message_types_by_name['TaskCollection'] = _TASKCOLLECTION
DESCRIPTOR.message_types_by_name['TaskSpec'] = _TASKSPEC
DESCRIPTOR.message_types_by_name['ScalarLinearRegressionTaskSpec'] = _SCALARLINEARREGRESSIONTASKSPEC
DESCRIPTOR.message_types_by_name['Scalar2LayerNNRegressionTaskSpec'] = _SCALAR2LAYERNNREGRESSIONTASKSPEC
DESCRIPTOR.message_types_by_name['ProjectedBinaryClassificationTask'] = _PROJECTEDBINARYCLASSIFICATIONTASK
DESCRIPTOR.message_types_by_name['ClassPair'] = _CLASSPAIR
DESCRIPTOR.message_types_by_name['ScalarLabelDataset'] = _SCALARLABELDATASET
DESCRIPTOR.message_types_by_name['FeatureVector'] = _FEATUREVECTOR
DESCRIPTOR.message_types_by_name['UnitTestFixedTask'] = _UNITTESTFIXEDTASK
DESCRIPTOR.message_types_by_name['UnitTestFixedTaskVector'] = _UNITTESTFIXEDTASKVECTOR
DESCRIPTOR.message_types_by_name['UnitTestZerosTaskSpec'] = _UNITTESTZEROSTASKSPEC
DESCRIPTOR.message_types_by_name['UnitTestOnesTaskSpec'] = _UNITTESTONESTASKSPEC
DESCRIPTOR.message_types_by_name['UnitTestIncrementTaskSpec'] = _UNITTESTINCREMENTTASKSPEC
DESCRIPTOR.enum_types_by_name['EvalType'] = _EVALTYPE
DESCRIPTOR.enum_types_by_name['ActivationType'] = _ACTIVATIONTYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TaskCollection = _reflection.GeneratedProtocolMessageType('TaskCollection', (_message.Message,), {
  'DESCRIPTOR' : _TASKCOLLECTION,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:automl_zero.TaskCollection)
  })
_sym_db.RegisterMessage(TaskCollection)

TaskSpec = _reflection.GeneratedProtocolMessageType('TaskSpec', (_message.Message,), {
  'DESCRIPTOR' : _TASKSPEC,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:automl_zero.TaskSpec)
  })
_sym_db.RegisterMessage(TaskSpec)

ScalarLinearRegressionTaskSpec = _reflection.GeneratedProtocolMessageType('ScalarLinearRegressionTaskSpec', (_message.Message,), {
  'DESCRIPTOR' : _SCALARLINEARREGRESSIONTASKSPEC,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:automl_zero.ScalarLinearRegressionTaskSpec)
  })
_sym_db.RegisterMessage(ScalarLinearRegressionTaskSpec)

Scalar2LayerNNRegressionTaskSpec = _reflection.GeneratedProtocolMessageType('Scalar2LayerNNRegressionTaskSpec', (_message.Message,), {
  'DESCRIPTOR' : _SCALAR2LAYERNNREGRESSIONTASKSPEC,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:automl_zero.Scalar2LayerNNRegressionTaskSpec)
  })
_sym_db.RegisterMessage(Scalar2LayerNNRegressionTaskSpec)

ProjectedBinaryClassificationTask = _reflection.GeneratedProtocolMessageType('ProjectedBinaryClassificationTask', (_message.Message,), {
  'DESCRIPTOR' : _PROJECTEDBINARYCLASSIFICATIONTASK,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:automl_zero.ProjectedBinaryClassificationTask)
  })
_sym_db.RegisterMessage(ProjectedBinaryClassificationTask)

ClassPair = _reflection.GeneratedProtocolMessageType('ClassPair', (_message.Message,), {
  'DESCRIPTOR' : _CLASSPAIR,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:automl_zero.ClassPair)
  })
_sym_db.RegisterMessage(ClassPair)

ScalarLabelDataset = _reflection.GeneratedProtocolMessageType('ScalarLabelDataset', (_message.Message,), {
  'DESCRIPTOR' : _SCALARLABELDATASET,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:automl_zero.ScalarLabelDataset)
  })
_sym_db.RegisterMessage(ScalarLabelDataset)

FeatureVector = _reflection.GeneratedProtocolMessageType('FeatureVector', (_message.Message,), {
  'DESCRIPTOR' : _FEATUREVECTOR,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:automl_zero.FeatureVector)
  })
_sym_db.RegisterMessage(FeatureVector)

UnitTestFixedTask = _reflection.GeneratedProtocolMessageType('UnitTestFixedTask', (_message.Message,), {
  'DESCRIPTOR' : _UNITTESTFIXEDTASK,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:automl_zero.UnitTestFixedTask)
  })
_sym_db.RegisterMessage(UnitTestFixedTask)

UnitTestFixedTaskVector = _reflection.GeneratedProtocolMessageType('UnitTestFixedTaskVector', (_message.Message,), {
  'DESCRIPTOR' : _UNITTESTFIXEDTASKVECTOR,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:automl_zero.UnitTestFixedTaskVector)
  })
_sym_db.RegisterMessage(UnitTestFixedTaskVector)

UnitTestZerosTaskSpec = _reflection.GeneratedProtocolMessageType('UnitTestZerosTaskSpec', (_message.Message,), {
  'DESCRIPTOR' : _UNITTESTZEROSTASKSPEC,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:automl_zero.UnitTestZerosTaskSpec)
  })
_sym_db.RegisterMessage(UnitTestZerosTaskSpec)

UnitTestOnesTaskSpec = _reflection.GeneratedProtocolMessageType('UnitTestOnesTaskSpec', (_message.Message,), {
  'DESCRIPTOR' : _UNITTESTONESTASKSPEC,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:automl_zero.UnitTestOnesTaskSpec)
  })
_sym_db.RegisterMessage(UnitTestOnesTaskSpec)

UnitTestIncrementTaskSpec = _reflection.GeneratedProtocolMessageType('UnitTestIncrementTaskSpec', (_message.Message,), {
  'DESCRIPTOR' : _UNITTESTINCREMENTTASKSPEC,
  '__module__' : 'task_pb2'
  # @@protoc_insertion_point(class_scope:automl_zero.UnitTestIncrementTaskSpec)
  })
_sym_db.RegisterMessage(UnitTestIncrementTaskSpec)


# @@protoc_insertion_point(module_scope)
