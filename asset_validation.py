import os
import re
import struct
from importlib import import_module
from types import SimpleNamespace
from typing import Any, cast

from unitypy_runtime import UnityPy

CompressionFlags = UnityPy.enums.BundleFile.CompressionFlags
CompressionHelper = UnityPy.helpers.CompressionHelper
SerializedType = import_module("UnityPy.files.SerializedFile").SerializedType

JsonDict = dict[str, Any]

_STRUCTURAL_VALIDATE_MAX_METADATA_BYTES = 64 * 1024 * 1024


def _align_value(value: int, alignment: int = 16) -> int:
    return int(value) + ((alignment - (int(value) % alignment)) % alignment)


def _read_c_string(handle: Any, limit: int = 4096) -> str:
    chunks = bytearray()
    while len(chunks) < limit:
        byte = handle.read(1)
        if not byte:
            raise ValueError("Unexpected EOF while reading C string")
        if byte == b"\0":
            return chunks.decode("utf-8", "surrogateescape")
        chunks.extend(byte)
    raise ValueError("C string exceeds limit")


def _parse_version_triplet(version_text: str) -> tuple[int, int, int]:
    match = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)", version_text or "")
    if not match:
        return (0, 0, 0)
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def _read_unityfs_structure(bundle_path: str) -> tuple[JsonDict | None, str | None]:
    try:
        with open(bundle_path, "rb") as handle:
            signature = _read_c_string(handle, limit=64)
            if signature != "UnityFS":
                return None, f"unsupported signature: {signature}"

            version = struct.unpack(">I", handle.read(4))[0]
            version_player = _read_c_string(handle)
            version_engine = _read_c_string(handle)
            total_file_size = struct.unpack(">Q", handle.read(8))[0]
            compressed_size, uncompressed_size, data_flags = struct.unpack(
                ">III", handle.read(12)
            )

            version_triplet = _parse_version_triplet(version_engine)
            uses_block_alignment = bool(
                version >= 7
                or (version_triplet[0] == 2019 and version_triplet >= (2019, 4, 15))
            )
            if uses_block_alignment:
                handle.seek(_align_value(handle.tell(), 16), os.SEEK_SET)

            blocks_info_start = handle.tell()
            file_length = os.path.getsize(bundle_path)
            if data_flags & 0x80:
                handle.seek(file_length - compressed_size, os.SEEK_SET)
                compressed_blocks_info = handle.read(compressed_size)
                data_start = blocks_info_start
            else:
                compressed_blocks_info = handle.read(compressed_size)
                data_start = blocks_info_start + compressed_size

            compression_flag = CompressionFlags(data_flags & 0x3F)
            blocks_info_bytes = cast(
                bytes,
                CompressionHelper.DECOMPRESSION_MAP[compression_flag](
                    compressed_blocks_info,
                    uncompressed_size,
                ),
            )
            if data_flags & 0x200:
                data_start = _align_value(data_start, 16)

            blocks_reader = UnityPy.streams.EndianBinaryReader(blocks_info_bytes)
            blocks_reader.read_bytes(16)
            block_count = blocks_reader.read_int()
            blocks: list[JsonDict] = []
            compressed_offset = 0
            uncompressed_offset = 0
            for _ in range(block_count):
                block_uncompressed = int(blocks_reader.read_u_int())
                block_compressed = int(blocks_reader.read_u_int())
                block_flags = int(blocks_reader.read_u_short())
                blocks.append(
                    {
                        "compressed_offset": compressed_offset,
                        "compressed_size": block_compressed,
                        "uncompressed_offset": uncompressed_offset,
                        "uncompressed_size": block_uncompressed,
                        "flags": block_flags,
                    }
                )
                compressed_offset += block_compressed
                uncompressed_offset += block_uncompressed

            directory_count = blocks_reader.read_int()
            directory_infos: list[JsonDict] = []
            for _ in range(directory_count):
                directory_infos.append(
                    {
                        "offset": int(blocks_reader.read_long()),
                        "size": int(blocks_reader.read_long()),
                        "flags": int(blocks_reader.read_u_int()),
                        "path": blocks_reader.read_string_to_null(),
                    }
                )

            return (
                {
                    "signature": signature,
                    "version": version,
                    "version_player": version_player,
                    "version_engine": version_engine,
                    "total_file_size": int(total_file_size),
                    "data_flags": int(data_flags),
                    "data_start": int(data_start),
                    "blocks": blocks,
                    "directory_infos": directory_infos,
                    "bundle_data_size": int(uncompressed_offset),
                },
                None,
            )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _read_unityfs_range(
    bundle_path: str,
    structure: JsonDict,
    offset: int,
    size: int,
) -> bytes:
    blocks = cast(list[JsonDict], structure.get("blocks") or [])
    data_start = int(structure.get("data_start") or 0)
    range_start = int(offset)
    range_end = range_start + int(size)
    out = bytearray()
    with open(bundle_path, "rb") as handle:
        for block in blocks:
            block_start = int(block["uncompressed_offset"])
            block_end = block_start + int(block["uncompressed_size"])
            if block_end <= range_start or block_start >= range_end:
                continue
            compression_flag = CompressionFlags(int(block["flags"]) & 0x3F)
            local_start = max(range_start, block_start) - block_start
            local_end = min(range_end, block_end) - block_start
            slice_len = local_end - local_start
            if slice_len <= 0:
                continue
            if compression_flag == CompressionFlags.NONE:
                handle.seek(
                    data_start + int(block["compressed_offset"]) + local_start,
                    os.SEEK_SET,
                )
                out.extend(handle.read(slice_len))
            else:
                handle.seek(data_start + int(block["compressed_offset"]), os.SEEK_SET)
                compressed_payload = handle.read(int(block["compressed_size"]))
                decompressed_payload = cast(
                    bytes,
                    CompressionHelper.DECOMPRESSION_MAP[compression_flag](
                        compressed_payload,
                        int(block["uncompressed_size"]),
                    ),
                )
                out.extend(decompressed_payload[local_start:local_end])
            if len(out) >= size:
                break
    return bytes(out[:size])


def _parse_serialized_header_info(
    header_bytes: bytes,
    entry_size: int,
) -> tuple[JsonDict | None, str | None]:
    if len(header_bytes) < 20:
        return None, "serialized header too small"
    metadata_size, file_size, version, data_offset = struct.unpack(
        ">4I", header_bytes[:16]
    )
    if version < 9:
        return None, f"unsupported serialized header version: {version}"
    endian = ">" if header_bytes[16] else "<"
    header_size = 20
    if version >= 22:
        if len(header_bytes) < 48:
            return None, "v22 serialized header too small"
        metadata_size = struct.unpack(">I", header_bytes[20:24])[0]
        file_size = struct.unpack(">Q", header_bytes[24:32])[0]
        data_offset = struct.unpack(">Q", header_bytes[32:40])[0]
        header_size = 48
    if metadata_size <= 0 or file_size <= 0 or data_offset <= 0:
        return None, "serialized header has non-positive critical fields"
    if file_size > entry_size:
        return None, f"serialized file_size {file_size} exceeds entry size {entry_size}"
    if data_offset > entry_size:
        return (
            None,
            f"serialized data_offset {data_offset} exceeds entry size {entry_size}",
        )
    return (
        {
            "metadata_size": int(metadata_size),
            "file_size": int(file_size),
            "version": int(version),
            "data_offset": int(data_offset),
            "endian": endian,
            "header_size": header_size,
        },
        None,
    )


def _is_probable_serialized_entry(header_bytes: bytes, entry_size: int) -> bool:
    info, _ = _parse_serialized_header_info(header_bytes, entry_size)
    return info is not None


def _parse_serialized_metadata_summary(
    metadata_bytes: bytes,
    entry_size: int,
) -> tuple[JsonDict | None, str | None]:
    header_info, reason = _parse_serialized_header_info(metadata_bytes[:48], entry_size)
    if header_info is None:
        return None, reason

    version = int(header_info["version"])
    endian = cast(str, header_info["endian"])
    data_offset = int(header_info["data_offset"])
    if len(metadata_bytes) < data_offset:
        return (
            None,
            f"metadata bytes truncated: need {data_offset}, got {len(metadata_bytes)}",
        )

    reader = UnityPy.streams.EndianBinaryReader(metadata_bytes, endian=endian)
    reader.Position = int(header_info["header_size"])

    unity_version = ""
    if version >= 7:
        unity_version = reader.read_string_to_null()
    if version >= 8:
        reader.read_int()
    enable_type_tree = True
    if version >= 13:
        enable_type_tree = bool(reader.read_boolean())

    type_count = int(reader.read_int())
    if type_count < 0 or type_count > 100000:
        return None, f"invalid type_count: {type_count}"

    dummy_file = SimpleNamespace(
        header=SimpleNamespace(version=version),
        _enable_type_tree=enable_type_tree,
    )
    for _ in range(type_count):
        SerializedType(reader, dummy_file, False)

    if 7 <= version < 14:
        reader.read_int()

    object_count = int(reader.read_int())
    if object_count <= 0:
        return None, f"invalid object_count: {object_count}"

    return (
        {
            "version": version,
            "unity_version": unity_version,
            "type_count": type_count,
            "object_count": object_count,
            "data_offset": data_offset,
        },
        None,
    )


def _structural_validate_unityfs_bundle(
    bundle_path: str,
    *,
    inner_names: list[str] | None = None,
) -> tuple[bool, str | None]:
    structure, reason = _read_unityfs_structure(bundle_path)
    if structure is None:
        return False, reason

    directory_infos = cast(list[JsonDict], structure.get("directory_infos") or [])
    if not directory_infos:
        return False, "bundle has no directory infos"

    bundle_data_size = int(structure.get("bundle_data_size") or 0)
    selected_paths = {
        str(name) for name in (inner_names or []) if str(name).strip()
    }
    selected_entries = [
        entry
        for entry in directory_infos
        if not selected_paths or str(entry.get("path")) in selected_paths
    ]
    if selected_paths and not selected_entries:
        return False, f"validation targets not found: {sorted(selected_paths)}"

    validated_serialized = 0
    for entry in selected_entries:
        entry_name = str(entry.get("path") or "")
        entry_offset = int(entry.get("offset") or 0)
        entry_size = int(entry.get("size") or 0)
        if entry_size <= 0:
            return False, f"entry has invalid size: {entry_name}"
        if entry_offset < 0 or entry_offset + entry_size > bundle_data_size:
            return False, f"entry range exceeds bundle data: {entry_name}"

        header_sample = _read_unityfs_range(
            bundle_path,
            structure,
            entry_offset,
            min(entry_size, 64),
        )
        if not _is_probable_serialized_entry(header_sample, entry_size):
            continue

        header_info, header_reason = _parse_serialized_header_info(
            header_sample, entry_size
        )
        if header_info is None:
            return False, f"{entry_name}: {header_reason}"

        metadata_span = int(header_info["data_offset"])
        if metadata_span > _STRUCTURAL_VALIDATE_MAX_METADATA_BYTES:
            return (
                False,
                f"{entry_name}: metadata span too large ({metadata_span} bytes)",
            )

        metadata_bytes = _read_unityfs_range(
            bundle_path,
            structure,
            entry_offset,
            metadata_span,
        )
        metadata_summary, metadata_reason = _parse_serialized_metadata_summary(
            metadata_bytes,
            entry_size,
        )
        if metadata_summary is None:
            return False, f"{entry_name}: {metadata_reason}"

        validated_serialized += 1

    if validated_serialized <= 0 and not selected_paths:
        return False, "no serialized entries validated"

    return True, None


def _collect_validation_inner_names(env_file: Any) -> list[str]:
    files = getattr(env_file, "files", None)
    if not isinstance(files, dict):
        return []
    names = [
        str(name)
        for name, value in files.items()
        if getattr(value, "is_changed", False)
    ]
    return sorted(set(names))
