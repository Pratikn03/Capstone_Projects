"""Pure-Python TFRecord and TensorFlow Example helpers.

The repo does not require TensorFlow for Stage 2 validation.  Waymo Motion
records are stored as serialized ``tf.train.Example`` payloads inside TFRecord
containers, so this module provides the minimal decode and test-write surface
needed by the validator and its unit tests.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

_LENGTH_STRUCT = struct.Struct("<Q")
_CRC_STRUCT = struct.Struct("<I")
_EXAMPLE_CLASS = None


def _masked_crc32c(payload: bytes) -> int:
    """Return the TFRecord masked CRC32C value when the backend is available."""
    try:
        import google_crc32c  # type: ignore
    except Exception:
        return 0
    crc = int(google_crc32c.value(payload))
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _build_example_class():
    global _EXAMPLE_CLASS
    if _EXAMPLE_CLASS is not None:
        return _EXAMPLE_CLASS

    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "tf_example_dynamic.proto"
    file_proto.package = "tensorflow"

    bytes_list = file_proto.message_type.add()
    bytes_list.name = "BytesList"
    field = bytes_list.field.add()
    field.name = "value"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_BYTES

    float_list = file_proto.message_type.add()
    float_list.name = "FloatList"
    field = float_list.field.add()
    field.name = "value"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT

    int64_list = file_proto.message_type.add()
    int64_list.name = "Int64List"
    field = int64_list.field.add()
    field.name = "value"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_INT64

    feature = file_proto.message_type.add()
    feature.name = "Feature"
    oneof = feature.oneof_decl.add()
    oneof.name = "kind"
    for number, name, type_name in (
        (1, "bytes_list", ".tensorflow.BytesList"),
        (2, "float_list", ".tensorflow.FloatList"),
        (3, "int64_list", ".tensorflow.Int64List"),
    ):
        field = feature.field.add()
        field.name = name
        field.number = number
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
        field.type_name = type_name
        field.oneof_index = 0

    features = file_proto.message_type.add()
    features.name = "Features"
    entry = features.nested_type.add()
    entry.name = "FeatureEntry"
    entry.options.map_entry = True
    key_field = entry.field.add()
    key_field.name = "key"
    key_field.number = 1
    key_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    key_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    value_field = entry.field.add()
    value_field.name = "value"
    value_field.number = 2
    value_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    value_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    value_field.type_name = ".tensorflow.Feature"

    field = features.field.add()
    field.name = "feature"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".tensorflow.Features.FeatureEntry"

    example = file_proto.message_type.add()
    example.name = "Example"
    field = example.field.add()
    field.name = "features"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".tensorflow.Features"

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_proto)
    _EXAMPLE_CLASS = message_factory.GetMessageClass(pool.FindMessageTypeByName("tensorflow.Example"))
    return _EXAMPLE_CLASS


def iter_tfrecord_records(path: str | Path, *, verify_crc: bool = False) -> Iterable[bytes]:
    """Yield serialized TF Example payloads from *path*."""
    file_path = Path(path)
    with file_path.open("rb") as handle:
        while True:
            header = handle.read(_LENGTH_STRUCT.size)
            if not header:
                break
            if len(header) != _LENGTH_STRUCT.size:
                raise ValueError(f"Truncated TFRecord length header in {file_path}")
            length = _LENGTH_STRUCT.unpack(header)[0]
            length_crc = handle.read(_CRC_STRUCT.size)
            if len(length_crc) != _CRC_STRUCT.size:
                raise ValueError(f"Truncated TFRecord length CRC in {file_path}")
            payload = handle.read(length)
            if len(payload) != length:
                raise ValueError(f"Truncated TFRecord payload in {file_path}")
            data_crc = handle.read(_CRC_STRUCT.size)
            if len(data_crc) != _CRC_STRUCT.size:
                raise ValueError(f"Truncated TFRecord data CRC in {file_path}")
            if verify_crc:
                observed_length_crc = _CRC_STRUCT.unpack(length_crc)[0]
                observed_data_crc = _CRC_STRUCT.unpack(data_crc)[0]
                expected_length_crc = _masked_crc32c(header)
                expected_data_crc = _masked_crc32c(payload)
                if observed_length_crc != expected_length_crc:
                    raise ValueError(f"Length CRC mismatch in {file_path}")
                if observed_data_crc != expected_data_crc:
                    raise ValueError(f"Payload CRC mismatch in {file_path}")
            yield payload


def parse_example_bytes(payload: bytes) -> dict[str, list[Any]]:
    """Decode one serialized ``tf.train.Example`` into Python lists."""
    example_cls = _build_example_class()
    example = example_cls()
    example.ParseFromString(payload)
    parsed: dict[str, list[Any]] = {}
    for key, feature in example.features.feature.items():
        kind = feature.WhichOneof("kind")
        if kind == "bytes_list":
            parsed[key] = list(feature.bytes_list.value)
        elif kind == "float_list":
            parsed[key] = [float(item) for item in feature.float_list.value]
        elif kind == "int64_list":
            parsed[key] = [int(item) for item in feature.int64_list.value]
        else:
            parsed[key] = []
    return parsed


def _normalize_feature_values(values: Any) -> list[Any]:
    if isinstance(values, bytes | str | int | float | bool):
        return [values]
    if isinstance(values, Sequence):
        return list(values)
    raise TypeError(f"Unsupported TF Example feature payload: {type(values)!r}")


def serialize_example_features(features: Mapping[str, Any]) -> bytes:
    """Serialize a feature mapping into a TensorFlow Example payload."""
    example_cls = _build_example_class()
    example = example_cls()
    for key, raw_values in features.items():
        values = _normalize_feature_values(raw_values)
        feature = example.features.feature[str(key)]
        if not values:
            continue
        first = values[0]
        if isinstance(first, bytes):
            feature.bytes_list.value.extend(values)
        elif isinstance(first, str):
            feature.bytes_list.value.extend(item.encode("utf-8") for item in values)
        elif isinstance(first, bool | int):
            feature.int64_list.value.extend(int(item) for item in values)
        else:
            feature.float_list.value.extend(float(item) for item in values)
    return example.SerializeToString()


def write_tfrecord_records(path: str | Path, payloads: Sequence[bytes]) -> Path:
    """Write serialized Example payloads to a TFRecord file."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as handle:
        for payload in payloads:
            length_bytes = _LENGTH_STRUCT.pack(len(payload))
            handle.write(length_bytes)
            handle.write(_CRC_STRUCT.pack(_masked_crc32c(length_bytes)))
            handle.write(payload)
            handle.write(_CRC_STRUCT.pack(_masked_crc32c(payload)))
    return file_path
