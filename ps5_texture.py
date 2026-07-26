import math
import os
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

# Initialize the repository's UnityPy selector before importing UnityPy enums.
from unitypy_runtime import UnityPy as _RuntimeUnityPy  # noqa: F401

try:
    from UnityPy.enums import TextureFormat as _UnityTextureFormatEnum
except Exception:  # pragma: no cover - optional at runtime
    _UnityTextureFormatEnum = None

try:
    import texture2ddecoder
except Exception:  # pragma: no cover - optional dependency
    texture2ddecoder = None


# KR: PS5 swizzle 비트 마스크: X축 인터리브 패턴
# EN: PS5 swizzle bit mask: X-axis interleave pattern
PS5_SWIZZLE_MASK_X = 0x385F0
# KR: PS5 swizzle 비트 마스크: Y축 인터리브 패턴
# EN: PS5 swizzle bit mask: Y-axis interleave pattern
PS5_SWIZZLE_MASK_Y = 0x07A0F
# KR: PS5 텍스처 회전 각도 (도)
# EN: PS5 texture rotation angle (degrees)
PS5_SWIZZLE_ROTATE = 90

# KR: PS5 텍스처 레이아웃 메타데이터.
#     word0: 첫 번째 qword 필드 (예: DXT1/BC1의 경우 0x1d0000000a).
#     block_pack: 블록당 바이트 수, 너비, 높이, 깊이를 하나의 정수로 패킹한 값.
# EN: PS5 texture layout metadata.
#     word0: first qword field (e.g. 0x1d0000000a for DXT1/BC1).
#     block_pack: packed integer of bytes per block, width, height, depth.
_PS5_LAYOUT_FORMAT_META: dict[int, dict[str, Any]] = {
    4: {"label": "R8B8G8A8", "word0": 0x00000004, "block_pack": 0x1010104},
    10: {"label": "DXT1|BC1", "word0": 0x1D0000000A, "block_pack": 0x1040408},
    12: {"label": "DXT5|BC3", "word0": 0x1D0000000C, "block_pack": 0x1040410},
    24: {"label": "BC6H", "word0": 0x1D00000018, "block_pack": 0x1040410},
    25: {"label": "BC7", "word0": 0x1D00000019, "block_pack": 0x1040410},
    26: {"label": "BC4", "word0": 0x1D0000001A, "block_pack": 0x1040408},
    27: {"label": "BC5", "word0": 0x1D0000001B, "block_pack": 0x1040410},
}

# KR: 런타임 레이아웃 선택 로직에서 사용하는 추가 포맷 플래그.
#     `layout_shift`는 다음과 같이 계산됨: ((flags & 0x6) * 2) + 8.
# EN: Additional format flags used in runtime layout selection logic.
#     `layout_shift` is computed as: ((flags & 0x6) * 2) + 8.
_PS5_LAYOUT_FORMAT_FLAGS: dict[int, int] = {
    10: 0x0024,  # DXT1|BC1
    12: 0x0000,  # DXT5|BC3
    24: 0x008C,  # BC6H
    25: 0x0094,  # BC7
    26: 0x00A4,  # BC4
    27: 0x0084,  # BC5
}
# KR: 각 텍스처 포맷에 대응하는 BC 디코더 함수명 매핑
# EN: BC decoder function name mapping for each texture format
_PS5_BC_DECODER_BY_FORMAT: dict[int, str] = {
    10: "decode_bc1",
    12: "decode_bc3",
    24: "decode_bc6",
    25: "decode_bc7",
    26: "decode_bc4",
    27: "decode_bc5",
}


def _ps5_unpack_block_pack(block_pack: int) -> tuple[int, int, int, int]:
    """KR: block_pack 정수에서 블록당 바이트 수, 블록 너비, 블록 높이, 깊이를 추출한다.
    매개변수:
        block_pack: 4바이트로 패킹된 정수 (하위부터 bytes_per_block, block_w, block_h, depth)
    반환값:
        (bytes_per_block, block_w, block_h, depth) 튜플

    EN: Extract bytes per block, block width, block height, and depth from a block_pack integer.
    Args:
        block_pack: 4바이트로 패킹된 정수 (하위부터 bytes_per_block, block_w, block_h, depth)
    Returns:
        (bytes_per_block, block_w, block_h, depth) 튜플
    """
    packed = int(block_pack) & 0xFFFFFFFF
    bytes_per_block = packed & 0xFF
    block_w = (packed >> 8) & 0xFF
    block_h = (packed >> 16) & 0xFF
    depth = (packed >> 24) & 0xFF
    return bytes_per_block, block_w, block_h, depth


def _ps5_build_bc_formats_from_layout_meta() -> dict[int, tuple[int, int, int, str]]:
    """KR: 레이아웃 메타데이터로부터 BC 포맷 정보 딕셔너리를 구성한다.
    반환값:
        {텍스처포맷ID: (블록너비, 블록높이, 블록당바이트, 디코더함수명)} 딕셔너리

    EN: Build a BC format info dictionary from layout metadata.
    Returns:
        {텍스처포맷ID: (블록너비, 블록높이, 블록당바이트, 디코더함수명)} 딕셔너리
    """
    out: dict[int, tuple[int, int, int, str]] = {}
    for texture_format, decoder_name in _PS5_BC_DECODER_BY_FORMAT.items():
        meta = _PS5_LAYOUT_FORMAT_META.get(int(texture_format))
        if not meta:
            continue
        bpb, bw, bh, depth = _ps5_unpack_block_pack(int(meta["block_pack"]))
        if depth != 1 or bpb <= 0 or bw <= 0 or bh <= 0:
            continue
        out[int(texture_format)] = (bw, bh, bpb, decoder_name)
    return out


# KR: BC 포맷 테이블: 레이아웃 메타에서 빌드된 {포맷ID: (블록W, 블록H, 블록바이트, 디코더명)}
# EN: BC format table: built from layout meta {formatID: (blockW, blockH, blockBytes, decoderName)}
_PS5_BC_FORMATS: dict[int, tuple[int, int, int, str]] = (
    _ps5_build_bc_formats_from_layout_meta()
)

# KR: Addrlib v2 (GFX10+) swizzle 모드 상수 (PS5에서 사용)
# EN: Addrlib v2 (GFX10+) swizzle mode constants (used on PS5)
_PS5_ADDR_SW_256B_S = 1  # KR: 256바이트 표준 / EN: 256-byte standard
_PS5_ADDR_SW_256B_D = 2  # KR: 256바이트 디스플레이 / EN: 256-byte display
_PS5_ADDR_SW_4KB_S = 5  # KR: 4KB 표준 / EN: 4KB standard
_PS5_ADDR_SW_4KB_D = 6  # KR: 4KB 디스플레이 / EN: 4KB display
_PS5_ADDR_SW_64KB_S = 9  # KR: 64KB 표준 / EN: 64KB standard
_PS5_ADDR_SW_64KB_D = 10  # KR: 64KB 디스플레이 / EN: 64KB display
_PS5_ADDR_SW_4KB_S_X = 21  # KR: 4KB 표준 확장 / EN: 4KB standard extended
_PS5_ADDR_SW_4KB_D_X = 22  # KR: 4KB 디스플레이 확장 / EN: 4KB display extended
_PS5_ADDR_SW_64KB_S_X = 25  # KR: 64KB 표준 확장 / EN: 64KB standard extended
_PS5_ADDR_SW_64KB_D_X = 26  # KR: 64KB 디스플레이 확장 / EN: 64KB display extended

# KR: BC 모드 정보: {모드명: (swizzle모드ID, 패턴정보테이블명, 로그2페이지크기, X확장여부)}
# EN: BC mode info: {modeName: (swizzleModeID, patternInfoTableName, log2PageSize, isXorMode)}
_PS5_BC_MODE_INFO: dict[str, tuple[int, str, int, bool]] = {
    "256B_S": (_PS5_ADDR_SW_256B_S, "GFX10_SW_256_S_PATINFO", 8, False),
    "256B_D": (_PS5_ADDR_SW_256B_D, "GFX10_SW_256_D_PATINFO", 8, False),
    "4KB_S": (_PS5_ADDR_SW_4KB_S, "GFX10_SW_4K_S_PATINFO", 12, False),
    "4KB_D": (_PS5_ADDR_SW_4KB_D, "GFX10_SW_4K_D_PATINFO", 12, False),
    "4KB_S_X": (_PS5_ADDR_SW_4KB_S_X, "GFX10_SW_4K_S_X_PATINFO", 12, True),
    "4KB_D_X": (_PS5_ADDR_SW_4KB_D_X, "GFX10_SW_4K_D_X_PATINFO", 12, True),
    "64KB_S": (_PS5_ADDR_SW_64KB_S, "GFX10_SW_64K_S_PATINFO", 16, False),
    "64KB_D": (_PS5_ADDR_SW_64KB_D, "GFX10_SW_64K_D_PATINFO", 16, False),
    "64KB_S_X": (_PS5_ADDR_SW_64KB_S_X, "GFX10_SW_64K_S_X_PATINFO", 16, True),
    "64KB_D_X": (_PS5_ADDR_SW_64KB_D_X, "GFX10_SW_64K_D_X_PATINFO", 16, True),
}
# KR: 빠른 모드 탐색 순서 (자주 사용되는 모드 우선)
# EN: Fast mode search order (frequently used modes first)
_PS5_BC_FAST_MODE_NAMES = ["4KB_S", "64KB_S", "4KB_D", "256B_S", "64KB_D", "256B_D"]

# KR: Thin 2D 타일 차원 (페이지 클래스별): {블록당바이트: (x비트수, y비트수)}
#     256바이트 페이지 클래스
# EN: Thin 2D tile dimensions (by page class): {bytesPerBlock: (xBits, yBits)}
#     256-byte page class
_PS5_LAYOUT_BLOCK256_2D_BITS: dict[int, tuple[int, int]] = {
    1: (4, 4),
    2: (4, 3),
    4: (3, 3),
    8: (3, 2),
    16: (2, 2),
}
# KR: 4KB 페이지 클래스
# EN: 4KB page class
_PS5_LAYOUT_BLOCK4K_2D_BITS: dict[int, tuple[int, int]] = {
    1: (6, 6),
    2: (6, 5),
    4: (5, 5),
    8: (5, 4),
    16: (4, 4),
}
# KR: 64KB 페이지 클래스
# EN: 64KB page class
_PS5_LAYOUT_BLOCK64K_2D_BITS: dict[int, tuple[int, int]] = {
    1: (8, 8),
    2: (8, 7),
    4: (7, 7),
    8: (7, 6),
    16: (6, 6),
}

# KR: 4KB_S 트리플릿: log2(블록당바이트)별 인덱싱 {블록바이트: (깊이비트, x비트, y비트)}
# EN: 4KB_S triplets: indexed by log2(bytesPerBlock) {blockBytes: (depthBits, xBits, yBits)}
_PS5_4KB_S_TRIPLETS_BY_BLOCK_BYTES: dict[int, tuple[int, int, int]] = {
    1: (0, 6, 6),
    2: (0, 6, 5),
    4: (0, 5, 5),
    8: (0, 5, 4),
    16: (0, 4, 4),
}

# KR: 포맷별 마이크로타일 차원: {블록당바이트: (x비트수, y비트수)}
# EN: Micro-tile dimensions per format: {bytesPerBlock: (xBits, yBits)}
_PS5_MICRO_TILE_BITS: dict[int, tuple[int, int]] = {
    1: (5, 4),  # KR: 32x16 픽셀 / EN: 32x16 pixels
    2: (4, 3),  # KR: 16x8 픽셀 / EN: 16x8 pixels
    3: (4, 3),  # KR: 16x8 픽셀 / EN: 16x8 pixels
    4: (3, 2),  # KR: 8x4 픽셀 / EN: 8x4 pixels
}
_PS5_MICRO_X_BITS_DEFAULT = 5  # KR: 8bpp 기본 X 비트 수 / EN: 8bpp default X bit count
_PS5_MICRO_Y_BITS_DEFAULT = 4  # KR: 8bpp 기본 Y 비트 수 / EN: 8bpp default Y bit count

# KR: 비정사각형 텍스처의 축 전치 여부: {블록당바이트: 전치여부}
# EN: Axis transpose for non-square textures: {bytesPerBlock: shouldTranspose}
_PS5_AXIS_TRANSPOSE: dict[int, bool] = {
    1: True,  # KR: Alpha8: 항상 전치 / EN: Alpha8: always transpose
    2: False,
    3: False,
    4: False,  # KR: RGBA32: 전치 안 함 / EN: RGBA32: no transpose
}


def _ps5_get_micro_tile_bits(bytes_per_element: int = 1) -> tuple[int, int]:
    """KR: 주어진 요소당 바이트 수에 대한 마이크로타일 비트 수를 반환한다.
    매개변수:
        bytes_per_element: 픽셀당 바이트 수 (bpe)
    반환값:
        (x_bits, y_bits) 튜플

    EN: Return micro-tile bit counts for the given bytes per element.
    Args:
        bytes_per_element: 픽셀당 바이트 수 (bpe)
    Returns:
        (x_bits, y_bits) 튜플
    """
    return _PS5_MICRO_TILE_BITS.get(
        bytes_per_element,
        (_PS5_MICRO_X_BITS_DEFAULT, _PS5_MICRO_Y_BITS_DEFAULT),
    )


@lru_cache(maxsize=64)
def compute_ps5_swizzle_masks(
    width: int,
    height: int,
    bytes_per_element: int = 1,
) -> tuple[int, int]:
    """KR: 텍스처 크기에 맞는 PS5 swizzle 비트 마스크를 계산한다.
    PS5 텍스처 메모리는 타일 기반 swizzle 레이아웃을 사용한다.
    마이크로타일 크기는 요소당 바이트 수(bpe)에 따라 결정된다:
      bpe=1 -> 32x16, bpe=4 -> 8x4 등.
    매크로타일 비트는 마이크로타일 비트 위에 다음 순서로 인터리브된다:
      첫 번째 Y, 첫 번째 X, 나머지 Y..., 나머지 X...
    매개변수:
        width: 텍스처 너비 (2의 거듭제곱이어야 함)
        height: 텍스처 높이 (2의 거듭제곱이어야 함)
        bytes_per_element: 픽셀당 바이트 수 (기본: 1)
    반환값:
        (mask_x, mask_y) swizzle 비트 마스크 튜플
    예외:
        ValueError: 유효하지 않은 차원이거나 2의 거듭제곱이 아닌 경우

    EN: Compute PS5 swizzle bit masks for the given texture dimensions.
    PS5 텍스처 메모리는 타일 기반 swizzle 레이아웃을 사용한다.
    마이크로타일 크기는 요소당 바이트 수(bpe)에 따라 결정된다:
      bpe=1 -> 32x16, bpe=4 -> 8x4 등.
    매크로타일 비트는 마이크로타일 비트 위에 다음 순서로 인터리브된다:
      첫 번째 Y, 첫 번째 X, 나머지 Y..., 나머지 X...
    Args:
        width: 텍스처 너비 (2의 거듭제곱이어야 함)
        height: 텍스처 높이 (2의 거듭제곱이어야 함)
        bytes_per_element: 픽셀당 바이트 수 (기본: 1)
    Returns:
        (mask_x, mask_y) swizzle 비트 마스크 튜플
    Raises:
        ValueError: 유효하지 않은 차원이거나 2의 거듭제곱이 아닌 경우
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"PS5 swizzle 마스크에 유효하지 않은 크기: {width}x{height}")
    if width & (width - 1) or height & (height - 1):
        raise ValueError(
            f"PS5 swizzle에는 2의 거듭제곱 크기가 필요합니다: {width}x{height}"
        )
    micro_x_bits, micro_y_bits = _ps5_get_micro_tile_bits(bytes_per_element)
    micro_w = 1 << micro_x_bits
    micro_h = 1 << micro_y_bits
    if width < micro_w or height < micro_h:
        raise ValueError(
            f"PS5 swizzle 마이크로타일({micro_w}x{micro_h})보다 작은 텍스처: "
            f"{width}x{height}"
        )
    total_x = width.bit_length() - 1  # log2(width)
    total_y = height.bit_length() - 1  # log2(height)
    macro_x = (
        total_x - micro_x_bits
    )  # KR: 매크로타일 X 비트 수 / EN: Number of macro-tile X bits
    macro_y = (
        total_y - micro_y_bits
    )  # KR: 매크로타일 Y 비트 수 / EN: Number of macro-tile Y bits

    mask_x = 0
    mask_y = 0
    pos = 0
    # KR: 마이크로타일 Y 비트 (최하위 비트부터 배치)
    # EN: Micro-tile Y bits (placed from least significant bit)
    for _ in range(micro_y_bits):
        mask_y |= 1 << pos
        pos += 1
    # KR: 마이크로타일 X 비트
    # EN: Micro-tile X bits
    for _ in range(micro_x_bits):
        mask_x |= 1 << pos
        pos += 1
    # KR: 매크로: 첫 번째 Y 비트
    # EN: Macro: first Y bit
    mx_rem = macro_x
    my_rem = macro_y
    if my_rem > 0:
        mask_y |= 1 << pos
        pos += 1
        my_rem -= 1
    # KR: 매크로: 첫 번째 X 비트
    # EN: Macro: first X bit
    if mx_rem > 0:
        mask_x |= 1 << pos
        pos += 1
        mx_rem -= 1
    # KR: 매크로: 나머지 Y 비트
    # EN: Macro: remaining Y bits
    for _ in range(my_rem):
        mask_y |= 1 << pos
        pos += 1
    # KR: 매크로: 나머지 X 비트
    # EN: Macro: remaining X bits
    for _ in range(mx_rem):
        mask_x |= 1 << pos
        pos += 1
    return mask_x, mask_y


def _ps5_dimensions_supported(
    width: int, height: int, bytes_per_element: int = 1
) -> bool:
    """KR: 주어진 텍스처 크기가 PS5 swizzle을 지원하는지 확인한다.
    매개변수:
        width: 텍스처 너비
        height: 텍스처 높이
        bytes_per_element: 픽셀당 바이트 수
    반환값:
        지원 가능하면 True, 아니면 False

    EN: Check whether the given texture dimensions support PS5 swizzle.
    Args:
        width: 텍스처 너비
        height: 텍스처 높이
        bytes_per_element: 픽셀당 바이트 수
    Returns:
        지원 가능하면 True, 아니면 False
    """
    if width <= 0 or height <= 0:
        return False
    if width & (width - 1) or height & (height - 1):
        return False
    xbits, ybits = _ps5_get_micro_tile_bits(bytes_per_element)
    micro_w = 1 << xbits
    micro_h = 1 << ybits
    return width >= micro_w and height >= micro_h


def _ps5_is_power_of_two(value: int) -> bool:
    """KR: 값이 2의 거듭제곱인지 확인한다.
    EN: Check if a value is a power of two.
    """
    return value > 0 and (value & (value - 1)) == 0


def _ps5_iter_divisor_pairs(total: int) -> Iterable[tuple[int, int]]:
    """KR: 주어진 정수의 모든 약수 쌍 (d, total/d)을 반복한다.
    매개변수:
        total: 약수를 구할 양의 정수
    반환값:
        (약수, 몲) 튜플의 반복자

    EN: Iterate over all divisor pairs (d, total/d) of a given integer.
    Args:
        total: 약수를 구할 양의 정수
    Returns:
        (약수, 몲) 튜플의 반복자
    """
    if total <= 0:
        return
    root = int(math.isqrt(total))
    for d in range(1, root + 1):
        if (total % d) != 0:
            continue
        q = total // d
        yield d, q
        if d != q:
            yield q, d


def _ps5_infer_physical_grid(
    total_elements: int,
    logical_width: int,
    logical_height: int,
    *,
    align_width: int,
    align_height: int,
) -> tuple[int, int]:
    """KR: 원시 요소 수로부터 유력한 물리적 그리드 크기를 추론한다.
    PS5 런타임은 표면(특히 BC 텍스처)을 논리적 크기 이상으로 패딩하는 경우가 많다.
    약수 쌍을 탐색하고 정렬/패딩 점수를 매겨 가장 적합한 물리적 WxH를 추론한다.
    매개변수:
        total_elements: 총 요소(픽셀/블록) 수
        logical_width: 논리적 텍스처 너비
        logical_height: 논리적 텍스처 높이
        align_width: 너비 정렬 단위
        align_height: 높이 정렬 단위
    반환값:
        (물리적너비, 물리적높이) 튜플

    EN: Infer the likely physical grid size from the raw element count.
    PS5 런타임은 표면(특히 BC 텍스처)을 논리적 크기 이상으로 패딩하는 경우가 많다.
    약수 쌍을 탐색하고 정렬/패딩 점수를 매겨 가장 적합한 물리적 WxH를 추론한다.
    Args:
        total_elements: 총 요소(픽셀/블록) 수
        logical_width: 논리적 텍스처 너비
        logical_height: 논리적 텍스처 높이
        align_width: 너비 정렬 단위
        align_height: 높이 정렬 단위
    Returns:
        (물리적너비, 물리적높이) 튜플
    """
    logical_total = max(0, logical_width) * max(0, logical_height)
    if (
        total_elements <= 0
        or logical_width <= 0
        or logical_height <= 0
        or total_elements < logical_total
    ):
        return logical_width, logical_height
    if total_elements == logical_total:
        return logical_width, logical_height

    best_pair: tuple[int, int] | None = None
    best_score: int | None = None

    for cand_w, cand_h in _ps5_iter_divisor_pairs(total_elements):
        if cand_w < logical_width or cand_h < logical_height:
            continue

        pad_w = cand_w - logical_width
        pad_h = cand_h - logical_height
        pad_area = (cand_w * cand_h) - logical_total

        # KR: 최소 여분 면적을 선호하고, 그 다음 높이 패딩보다 너비 패딩을 선호
        # EN: Prefer minimal excess area, then prefer width padding over height padding
        score = pad_area * 1000 + pad_h * 32 + pad_w * 4
        if align_width > 1 and (cand_w % align_width) != 0:
            score += 250  # KR: 너비 정렬 미달 페널티 / EN: Width alignment miss penalty
        if align_height > 1 and (cand_h % align_height) != 0:
            score += (
                250  # KR: 높이 정렬 미달 페널티 / EN: Height alignment miss penalty
            )
        if _ps5_is_power_of_two(cand_w):
            score -= 32  # KR: 2의 거듭제곱 너비 보너스 / EN: Power-of-two width bonus
        if _ps5_is_power_of_two(cand_h):
            score -= 16  # KR: 2의 거듭제곱 높이 보너스 / EN: Power-of-two height bonus

        if best_score is None or score < best_score:
            best_score = score
            best_pair = (cand_w, cand_h)

    return best_pair if best_pair is not None else (logical_width, logical_height)


def _ps5_align_up(value: int, align: int) -> int:
    """KR: 값을 지정된 정렬 단위로 올림 정렬한다.
    EN: Align a value up to the specified alignment unit.
    """
    if align <= 1:
        return int(value)
    return ((int(value) + int(align) - 1) // int(align)) * int(align)


def _ps5_physical_grid_candidates_for_mode(
    total_elements: int,
    logical_width: int,
    logical_height: int,
    *,
    bytes_per_block: int,
    mode_name: str,
    align_width: int,
    align_height: int,
) -> list[tuple[int, int]]:
    """KR: 주어진 BC 스위즐 모드에 대한 물리 그리드 후보를 우선순위 순서로 반환한다.
    EN: Return physical grid candidates in priority order for a given BC swizzle mode.

    순서:
    1) 레이아웃 정렬된 후보,
    2) 약수 기반 추론 후보,
    3) 논리 그리드.
    """
    out: list[tuple[int, int]] = []

    def _push(pair: tuple[int, int]) -> None:
        # KR: 유효하지 않은 후보 필터링: 양수, 논리 크기 이상, 총 요소 수 이하
        # EN: Filter out invalid candidates: positive, at least logical size, within total elements
        if pair[0] <= 0 or pair[1] <= 0:
            return
        if pair[0] < logical_width or pair[1] < logical_height:
            return
        if pair[0] * pair[1] > total_elements:
            return
        if pair not in out:
            out.append(pair)

    # KR: 타일 비트 차원에서 정렬된 후보 계산
    # EN: Compute aligned candidates from tile bit dimensions
    bits = _ps5_tile_bit_dimensions_for_mode(mode_name, bytes_per_block)
    if bits is not None:
        tile_w = (
            1 << bits[0]
        )  # KR: 타일 너비 (2의 거듭제곱) / EN: Tile width (power of two)
        tile_h = (
            1 << bits[1]
        )  # KR: 타일 높이 (2의 거듭제곱) / EN: Tile height (power of two)
        aligned_w = _ps5_align_up(logical_width, tile_w)
        aligned_h = _ps5_align_up(logical_height, tile_h)
        _push((aligned_w, aligned_h))
        # KR: 정렬된 너비로 총 요소 수를 나누어 높이 후보 추론
        # EN: Infer height candidate by dividing total elements by aligned width
        if aligned_w > 0 and (total_elements % aligned_w) == 0:
            aligned_h_from_total = total_elements // aligned_w
            if (
                aligned_h_from_total >= aligned_h
                and (aligned_h_from_total % tile_h) == 0
            ):
                _push((aligned_w, aligned_h_from_total))
        # KR: 정렬된 높이로 총 요소 수를 나누어 너비 후보 추론
        # EN: Infer width candidate by dividing total elements by aligned height
        if aligned_h > 0 and (total_elements % aligned_h) == 0:
            aligned_w_from_total = total_elements // aligned_h
            if (
                aligned_w_from_total >= aligned_w
                and (aligned_w_from_total % tile_w) == 0
            ):
                _push((aligned_w_from_total, aligned_h))

    # KR: 약수 기반 물리 그리드 추론 결과 추가
    # EN: Add divisor-based physical grid inference result
    inferred = _ps5_infer_physical_grid(
        total_elements,
        logical_width,
        logical_height,
        align_width=align_width,
        align_height=align_height,
    )
    _push(inferred)
    # KR: 최종 폴백: 논리 그리드 자체
    # EN: Final fallback: the logical grid itself
    _push((logical_width, logical_height))
    return out


def _ps5_read_lines(path: Path) -> list[str]:
    """KR: 파일을 UTF-8로 읽어 줄 단위 리스트로 반환한다.
    EN: Read a file as UTF-8 and return a list of lines.
    """
    return path.read_text(encoding="utf-8").splitlines()


def _ps5_extract_block(lines: list[str], decl_prefix: str) -> list[str]:
    """KR: C 헤더 파일에서 선언 접두사로 시작하는 블록의 본문 줄들을 추출한다.
    EN: Extract body lines of a block starting with a declaration prefix from a C header file.
    """
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(decl_prefix):
            # KR: 선언부 다음 줄부터 본문 시작
            # EN: Body starts from the line after the declaration
            start = i + 1
            break
    if start is None:
        raise RuntimeError(f"Declaration not found: {decl_prefix}")
    out: list[str] = []
    for line in lines[start:]:
        # KR: 닫는 중괄호를 만나면 블록 종료
        # EN: Block ends at closing brace
        if line.strip().startswith("};"):
            break
        out.append(line)
    return out


def _ps5_expr_to_mask(expr: str) -> int:
    """KR: 스위즐 패턴 수식 문자열(예: "X0^Y1^Z2")을 64비트 마스크 정수로 변환한다.
    EN: Convert a swizzle pattern expression string (e.g. "X0^Y1^Z2") to a 64-bit mask integer.

    각 채널(X/Y/Z/S)은 16비트 영역을 차지하며, XOR로 결합된다.
    """
    expr = expr.strip()
    if expr == "0":
        return 0
    total = 0
    # KR: "^"로 분리된 각 토큰을 파싱하여 비트 마스크에 XOR 누적
    # EN: Parse each token separated by '^' and XOR-accumulate into the bit mask
    for part in expr.split("^"):
        token = part.strip()
        if not token:
            continue
        ch = token[0]  # KR: 채널 문자: X, Y, Z, S / EN: Channel character: X, Y, Z, S
        if ch not in "XYZS" or not token[1:].isdigit():
            raise RuntimeError(f"Unexpected token in swizzle expression: {token}")
        idx = int(token[1:])  # KR: 비트 인덱스 (0~15) / EN: Bit index (0~15)
        if idx < 0 or idx > 15:
            raise RuntimeError(f"Token bit out of range: {token}")
        chan = "XYZS".index(
            ch
        )  # KR: 채널 오프셋 (X=0, Y=1, Z=2, S=3) / EN: Channel offset (X=0, Y=1, Z=2, S=3)
        # KR: 채널별 16비트 슬롯 내의 해당 비트를 설정
        # EN: Set the corresponding bit within the channel's 16-bit slot
        total ^= 1 << (chan * 16 + idx)
    return total


def _ps5_parse_nibble_array(
    lines: list[str], name: str, row_width: int
) -> list[list[int]]:
    """KR: gfx10SwizzlePattern.h에서 니블(nibble) 배열을 파싱하여 마스크 행렬로 반환한다.
    EN: Parse a nibble array from gfx10SwizzlePattern.h and return it as a mask matrix.
    """
    block = _ps5_extract_block(lines, f"const UINT_64 {name}")
    rows: list[list[int]] = []
    for line in block:
        if "{" not in line:
            continue
        # KR: 중괄호 내부의 수식 항목들을 추출
        # EN: Extract expression items inside braces
        body = line.split("{", 1)[1].split("}", 1)[0]
        items = [x.strip() for x in body.split(",") if x.strip()]
        if len(items) < row_width:
            continue
        # KR: 각 수식을 64비트 마스크로 변환
        # EN: Convert each expression to a 64-bit mask
        rows.append([_ps5_expr_to_mask(items[i]) for i in range(row_width)])
    return rows


def _ps5_parse_patinfo_array(
    lines: list[str], name: str
) -> list[tuple[int, int, int, int, int]]:
    """KR: ADDR_SW_PATINFO 배열을 파싱하여 (maxItemCount, idx01, idx2, idx3, idx4) 튜플 리스트로 반환한다.
    EN: Parse an ADDR_SW_PATINFO array and return a list of (maxItemCount, idx01, idx2, idx3, idx4) tuples.
    """
    block = _ps5_extract_block(lines, f"const ADDR_SW_PATINFO {name}")
    rows: list[tuple[int, int, int, int, int]] = []
    # KR: 5개 정수 필드를 가진 구조체 초기화 패턴 매칭
    # EN: Pattern matching for struct initializer with 5 integer fields
    pat = re.compile(
        r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,?\s*\}"
    )
    for line in block:
        m = pat.search(line)
        if m:
            rows.append(tuple(int(m.group(i)) for i in range(1, 6)))
    return rows


@lru_cache(maxsize=1)
def _ps5_resolve_swizzle_pattern_path() -> str | None:
    """KR: gfx10SwizzlePattern.h 파일 경로를 환경변수 또는 기본 위치에서 탐색하여 반환한다.
    EN: Search for the gfx10SwizzlePattern.h file path from environment variable or default locations.
    """
    # KR: 환경변수 PS5_SWIZZLE_PATTERN_H를 우선 확인
    # EN: Check PS5_SWIZZLE_PATTERN_H environment variable first
    env_path = os.environ.get("PS5_SWIZZLE_PATTERN_H")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    # KR: 리포지토리 내부의 기본 Addrlib 헤더 경로
    # EN: Default Addrlib header path inside the repository
    repo_root = Path(__file__).resolve().parent
    candidates.append(
        repo_root
        / "TMP_Info"
        / "method1"
        / "pal"
        / "src"
        / "core"
        / "imported"
        / "addrlib"
        / "src"
        / "gfx10"
        / "gfx10SwizzlePattern.h"
    )
    # KR: 후보 경로 중 존재하는 첫 번째 파일 반환
    # EN: Return the first existing file among candidates
    for path in candidates:
        if path.exists():
            return str(path)
    return None


@lru_cache(maxsize=1)
def _ps5_load_bc_pattern_tables() -> dict[str, Any] | None:
    """KR: gfx10SwizzlePattern.h에서 모든 니블 및 패턴 정보 테이블을 로드한다.
    EN: Load all nibble and pattern info tables from gfx10SwizzlePattern.h.

    파일이 없거나 파싱 실패 시 None을 반환한다.
    """
    pattern_path = _ps5_resolve_swizzle_pattern_path()
    if not pattern_path:
        return None
    try:
        lines = _ps5_read_lines(Path(pattern_path))
        # KR: 니블 배열 파싱 (nib01은 8열, nib2/3/4는 4열)
        # EN: Parse nibble arrays (nib01 has 8 columns, nib2/3/4 have 4 columns)
        nib01 = _ps5_parse_nibble_array(lines, "GFX10_SW_PATTERN_NIBBLE01", 8)
        nib2 = _ps5_parse_nibble_array(lines, "GFX10_SW_PATTERN_NIBBLE2", 4)
        nib3 = _ps5_parse_nibble_array(lines, "GFX10_SW_PATTERN_NIBBLE3", 4)
        nib4 = _ps5_parse_nibble_array(lines, "GFX10_SW_PATTERN_NIBBLE4", 4)
        # KR: 각 스위즐 모드별 PATINFO 배열 파싱
        # EN: Parse PATINFO array for each swizzle mode
        patinfo_tables = {
            mode_name: _ps5_parse_patinfo_array(lines, info[1])
            for mode_name, info in _PS5_BC_MODE_INFO.items()
        }
        return {
            "nib01": nib01,
            "nib2": nib2,
            "nib3": nib3,
            "nib4": nib4,
            "patinfo_tables": patinfo_tables,
        }
    except Exception:
        return None


def _ps5_compute_thin_block_dim(
    block_bits: int, bytes_per_block: int
) -> tuple[int, int]:
    """KR: Thin 2D 블록의 너비/높이 차원을 계산한다 (addrlib2.cpp::ComputeThinBlockDimension, numSamples=1).
    EN: Compute thin 2D block width/height dimensions (addrlib2.cpp::ComputeThinBlockDimension, numSamples=1).
    """
    log2_ele = int(
        math.log2(bytes_per_block)
    )  # KR: 요소당 바이트의 log2 / EN: log2 of bytes per element
    log2_num_ele = (
        block_bits - log2_ele
    )  # KR: 블록 내 요소 수의 log2 / EN: log2 of number of elements in block
    log2_w = (
        (log2_num_ele + 1) // 2
    )  # KR: 너비 비트 수 (높이보다 1비트 많거나 같음) / EN: Width bits (equal to or 1 more than height bits)
    w = 1 << log2_w
    h = 1 << (log2_num_ele - log2_w)
    return w, h


def _ps5_parity(value: int) -> int:
    """KR: 정수의 설정된 비트 수의 패리티(홀짝)를 반환한다 (0 또는 1).
    EN: Return the parity (even/odd) of set bits in an integer (0 or 1).
    """
    return value.bit_count() & 1


def _ps5_tile_bit_dimensions_for_mode(
    mode_name: str,
    bytes_per_block: int,
) -> tuple[int, int] | None:
    """KR: 레이아웃 테이블에서 thin-2D 타일의 비트 차원(x_bits, y_bits)을 조회한다.
    EN: Look up thin-2D tile bit dimensions (x_bits, y_bits) from the layout table.
    """
    if mode_name.endswith("_X"):
        # KR: XOR 스위즐 변형은 여기서 재구성할 수 없는 추가 방정식 비트가 필요하다.
        # EN: XOR swizzle variants require additional equation bits that cannot be reconstructed here.
        return None
    table: dict[int, tuple[int, int]] | None = None
    # KR: 페이지 크기별 레이아웃 테이블 선택
    # EN: Select layout table by page size
    if mode_name.startswith("256B_"):
        table = _PS5_LAYOUT_BLOCK256_2D_BITS
    elif mode_name.startswith("4KB_"):
        table = _PS5_LAYOUT_BLOCK4K_2D_BITS
    elif mode_name.startswith("64KB_"):
        table = _PS5_LAYOUT_BLOCK64K_2D_BITS
    if table is None:
        return None
    return table.get(int(bytes_per_block))


def _ps5_tile_bit_order_for_mode(mode_name: str, bytes_per_block: int) -> str:
    """KR: BC 레이아웃 폴백용 타일 내부 비트 인터리빙 순서를 선택한다.
    EN: Select the in-tile bit interleaving order for BC layout fallback.
    """
    if mode_name.startswith(("4KB_", "64KB_")):
        if int(bytes_per_block) >= 16:
            return "yxyx"  # KR: Y/X 비트 교차 인터리빙 / EN: Y/X bit interleaved alternation
        if int(bytes_per_block) == 8:
            return "x0_yxyx"  # KR: X의 최하위 비트 선행 후 Y/X 교차 / EN: X LSB precedes, then Y/X alternation
    return "yx"  # KR: 기본: Y 하위, X 상위 / EN: Default: Y lower, X upper


def _ps5_local_swizzle_index(
    local_x: int,
    local_y: int,
    x_bits: int,
    y_bits: int,
    order: str,
) -> int:
    """KR: 타일 내부 (local_x, local_y) 좌표를 지정된 비트 순서에 따라 선형 인덱스로 변환한다.
    EN: Convert tile-local (local_x, local_y) coordinates to a linear index according to the specified bit order.
    """
    if order == "yx":
        # KR: 단순 순서: Y가 하위 비트, X가 상위 비트
        # EN: Simple order: Y is lower bits, X is upper bits
        return local_y + (local_x << y_bits)
    if order == "yxyx":
        # KR: Y/X 비트를 번갈아 인터리빙
        # EN: Y/X bits interleaved alternately
        out = 0
        bit_pos = 0
        for bit in range(max(x_bits, y_bits)):
            if bit < y_bits:
                # KR: Y의 bit번째 비트를 출력 위치에 배치
                # EN: Place Y's bit-th bit at output position
                out |= ((local_y >> bit) & 1) << bit_pos
                bit_pos += 1
            if bit < x_bits:
                # KR: X의 bit번째 비트를 출력 위치에 배치
                # EN: Place X's bit-th bit at output position
                out |= ((local_x >> bit) & 1) << bit_pos
                bit_pos += 1
        return out
    if order == "x0_yxyx":
        # KR: X의 최하위 비트가 먼저 오고, 나머지는 Y/X 교차 인터리빙
        # EN: X's least significant bit comes first, rest are Y/X alternating interleave
        if x_bits <= 0:
            return local_y
        out = local_x & 1  # KR: X의 bit0을 최하위에 배치 / EN: Place X's bit0 at LSB
        bit_pos = 1
        for bit in range(max(x_bits - 1, y_bits)):
            if bit < y_bits:
                out |= ((local_y >> bit) & 1) << bit_pos
                bit_pos += 1
            if bit < (x_bits - 1):
                # KR: X의 bit1 이상을 인터리빙
                # EN: Interleave X's bit1 and above
                out |= ((local_x >> (bit + 1)) & 1) << bit_pos
                bit_pos += 1
        return out
    # KR: 기본 폴백: yx 순서
    # EN: Default fallback: yx order
    return local_y + (local_x << y_bits)


def _ps5_4kb_s_scalar_mix_bytes1(value: int) -> int:
    """KR: 4KB_S 경로에서 1바이트 블록용 스칼라 믹스 헬퍼.
    EN: Scalar mix helper for 1-byte blocks in 4KB_S path.
    """
    v = int(value)
    # KR: 비트 시프트 후 마스크로 특정 비트 위치만 추출하여 XOR 결합
    # EN: Extract specific bit positions via shift and mask, then XOR combine
    return ((v << 4) & 0x1F0) ^ ((v << 5) & 0x400)


def _ps5_4kb_s_scalar_mix_bytes2_4(value: int) -> int:
    """KR: 4KB_S 경로에서 2바이트 또는 4바이트 블록용 스칼라 믹스 헬퍼.
    EN: Scalar mix helper for 2-byte or 4-byte blocks in 4KB_S path.
    """
    v = int(value)
    return ((v << 4) & 0x70) ^ ((v << 5) & 0x100) ^ ((v << 6) & 0x400)


def _ps5_4kb_s_scalar_mix_bytes8_16(value: int) -> int:
    """KR: 4KB_S 경로에서 8바이트 또는 16바이트 블록용 스칼라 믹스 헬퍼.
    EN: Scalar mix helper for 8-byte or 16-byte blocks in 4KB_S path.
    """
    v = int(value)
    return ((v << 4) & 0x30) ^ ((v << 6) & 0x100) ^ ((v << 7) & 0x400)


def _ps5_4kb_s_vector_mix_bytes1(value: int) -> int:
    """KR: 4KB_S 경로에서 1바이트 블록용 벡터 믹스 헬퍼.
    EN: Vector mix helper for 1-byte blocks in 4KB_S path.
    """
    v = int(value)
    # KR: 하위 4비트 직접 사용 + 상위 비트를 시프트하여 XOR 결합
    # EN: Use lower 4 bits directly + shift upper bits for XOR combination
    return (v & 0x0F) ^ ((v << 5) & 0x200) ^ ((v << 6) & 0x800)


def _ps5_4kb_s_vector_mix_bytes2(value: int) -> int:
    """KR: 4KB_S 경로에서 2바이트 블록용 벡터 믹스 헬퍼.
    EN: Vector mix helper for 2-byte blocks in 4KB_S path.
    """
    v = int(value)
    return (
        ((v << 1) & 0x0E) ^ ((v << 4) & 0x80) ^ ((v << 5) & 0x200) ^ ((v << 6) & 0x800)
    )


def _ps5_4kb_s_vector_mix_bytes4(value: int) -> int:
    """KR: 4KB_S 경로에서 4바이트 블록용 벡터 믹스 헬퍼.
    EN: Vector mix helper for 4-byte blocks in 4KB_S path.
    """
    v = int(value)
    return (
        ((v << 2) & 0x0C) ^ ((v << 5) & 0x80) ^ ((v << 6) & 0x200) ^ ((v << 7) & 0x800)
    )


def _ps5_4kb_s_vector_mix_bytes8(value: int) -> int:
    """KR: 4KB_S 경로에서 8바이트 블록용 벡터 믹스 헬퍼.
    EN: Vector mix helper for 8-byte blocks in 4KB_S path.
    """
    v = int(value)
    return (
        ((v << 3) & 0x08) ^ ((v << 5) & 0xC0) ^ ((v << 6) & 0x200) ^ ((v << 7) & 0x800)
    )


def _ps5_4kb_s_vector_mix_bytes16(value: int) -> int:
    """KR: 4KB_S 경로에서 16바이트 블록용 벡터 믹스 헬퍼.
    EN: Vector mix helper for 16-byte blocks in 4KB_S path.
    """
    v = int(value)
    return ((v << 6) & 0xC0) ^ ((v << 7) & 0x200) ^ ((v << 8) & 0x800)


def _ps5_4kb_s_tile_index(
    local_x: int,
    local_y: int,
    bytes_per_block: int,
) -> int | None:
    """KR: 4KB_S 경로에서 타일 내부 좌표를 스위즐된 인덱스로 변환한다.
    EN: Convert tile-local coordinates to a swizzled index in 4KB_S path.
    """
    bpb = int(bytes_per_block)
    # KR: 블록 크기별로 적절한 스칼라(Y)/벡터(X) 믹스 함수 선택
    # EN: Select appropriate scalar(Y)/vector(X) mix function by block size
    if bpb == 1:
        base = _ps5_4kb_s_scalar_mix_bytes1(local_y)
        mixed = base ^ _ps5_4kb_s_vector_mix_bytes1(local_x)
    elif bpb == 2:
        base = _ps5_4kb_s_scalar_mix_bytes2_4(local_y)
        mixed = base ^ _ps5_4kb_s_vector_mix_bytes2(local_x)
    elif bpb == 4:
        base = _ps5_4kb_s_scalar_mix_bytes2_4(local_y)
        mixed = base ^ _ps5_4kb_s_vector_mix_bytes4(local_x)
    elif bpb == 8:
        base = _ps5_4kb_s_scalar_mix_bytes8_16(local_y)
        mixed = base ^ _ps5_4kb_s_vector_mix_bytes8(local_x)
    elif bpb == 16:
        base = _ps5_4kb_s_scalar_mix_bytes8_16(local_y)
        mixed = base ^ _ps5_4kb_s_vector_mix_bytes16(local_x)
    else:
        return None
    # KR: 바이트 오프셋을 요소 인덱스로 변환 (블록 크기만큼 우측 시프트)
    # EN: Convert byte offset to element index (right shift by block size)
    return mixed >> int(math.log2(bpb))


def _ps5_build_bc_lut_from_layout_rules(
    block_w: int,
    block_h: int,
    bytes_per_block: int,
    mode_name: str,
    pipe_bank_xor: int,
) -> tuple[int, ...] | None:
    """KR: 외부 패턴 헤더 없이 레이아웃 규칙만으로 BC LUT를 구축한다.
    EN: Build a BC LUT using only layout rules without external pattern headers.

    pipe_bank_xor가 0이 아니면 이 경로는 지원하지 않으므로 None을 반환한다.
    """
    if pipe_bank_xor != 0:
        return None
    bits = _ps5_tile_bit_dimensions_for_mode(mode_name, bytes_per_block)
    if bits is None:
        return None
    x_bits, y_bits = bits
    if x_bits <= 0 or y_bits <= 0:
        return None

    tile_w = 1 << x_bits  # KR: 타일 너비 (블록 단위) / EN: Tile width (in blocks)
    tile_h = 1 << y_bits  # KR: 타일 높이 (블록 단위) / EN: Tile height (in blocks)
    if tile_w <= 0 or tile_h <= 0 or block_w <= 0 or block_h <= 0:
        return None

    local_order = _ps5_tile_bit_order_for_mode(mode_name, bytes_per_block)
    use_4kb_s_helper_formula = mode_name == "4KB_S"
    macro_cols = (
        block_w + tile_w - 1
    ) // tile_w  # KR: 매크로 타일 열 수 / EN: Number of macro tile columns
    tile_elements = (
        tile_w * tile_h
    )  # KR: 타일 하나의 총 요소 수 / EN: Total elements per tile
    total = block_w * block_h

    lut: list[int] = [0] * total
    for y in range(block_h):
        macro_y = y // tile_h  # KR: 매크로 타일 행 인덱스 / EN: Macro tile row index
        local_y = y & (tile_h - 1)  # KR: 타일 내부 Y 좌표 / EN: Tile-local Y coordinate
        row_base = y * block_w
        macro_row_base = macro_y * macro_cols * tile_elements
        for x in range(block_w):
            macro_x = (
                x // tile_w
            )  # KR: 매크로 타일 열 인덱스 / EN: Macro tile column index
            local_x = x & (
                tile_w - 1
            )  # KR: 타일 내부 X 좌표 / EN: Tile-local X coordinate
            if use_4kb_s_helper_formula:
                # KR: 4KB_S 모드: 전용 비트 믹스 공식 사용
                # EN: 4KB_S mode: use dedicated bit mix formula
                local_off = _ps5_4kb_s_tile_index(
                    local_x,
                    local_y,
                    bytes_per_block,
                )
                if local_off is None:
                    return None
            else:
                # KR: 일반 모드: 비트 인터리빙 순서에 따라 인덱스 계산
                # EN: General mode: compute index according to bit interleaving order
                local_off = _ps5_local_swizzle_index(
                    local_x,
                    local_y,
                    x_bits,
                    y_bits,
                    local_order,
                )
            if local_off < 0 or local_off >= tile_elements:
                return None
            # KR: 매크로 타일 기준 오프셋 + 타일 내 로컬 오프셋 = 스위즐된 인덱스
            # EN: Macro tile base offset + tile-local offset = swizzled index
            swizzled_idx = macro_row_base + macro_x * tile_elements + local_off
            if swizzled_idx >= total:
                return None
            lut[row_base + x] = swizzled_idx
    return tuple(lut)


def _ps5_compute_offset(
    pattern_bits: list[int],
    block_bits: int,
    x: int,
    y: int,
    z: int = 0,
    s: int = 0,
) -> int:
    """KR: Addrlib 패턴 비트 배열로부터 (x, y, z, s) 좌표의 블록 내 오프셋을 계산한다.
    EN: Compute the in-block offset for (x, y, z, s) coordinates from an Addrlib pattern bit array.

    각 출력 비트는 X/Y/Z/S 채널 마스크의 패리티를 XOR 결합하여 결정된다.
    """
    out = 0
    for i in range(block_bits):
        m = pattern_bits[i]
        if m == 0:
            continue
        # KR: 64비트 마스크에서 각 채널(X/Y/Z/S)의 16비트 슬롯 추출
        # EN: Extract 16-bit slot for each channel (X/Y/Z/S) from 64-bit mask
        xmask = m & 0xFFFF
        ymask = (m >> 16) & 0xFFFF
        zmask = (m >> 32) & 0xFFFF
        smask = (m >> 48) & 0xFFFF
        # KR: 각 채널의 좌표와 마스크를 AND한 뒤 패리티를 XOR 결합
        # EN: AND coordinate with mask for each channel, then XOR combine parities
        bit = (
            _ps5_parity(x & xmask)
            ^ _ps5_parity(y & ymask)
            ^ _ps5_parity(z & zmask)
            ^ _ps5_parity(s & smask)
        )
        out |= (
            bit << i
        )  # KR: 결과 비트를 출력 오프셋의 i번째 위치에 배치 / EN: Place result bit at i-th position of output offset
    return out


def _ps5_build_full_pattern(
    nib01: list[list[int]],
    nib2: list[list[int]],
    nib3: list[list[int]],
    nib4: list[list[int]],
    patinfo: tuple[int, int, int, int, int],
) -> list[int]:
    """KR: PATINFO 인덱스를 사용하여 니블 테이블 4개를 연결한 완전한 패턴 비트 배열을 구축한다.
    EN: Build a complete pattern bit array by concatenating 4 nibble tables using PATINFO indices.
    """
    _, idx01, idx2, idx3, idx4 = patinfo
    # KR: 각 니블 테이블에 대한 인덱스 범위 검증
    # EN: Validate index range for each nibble table
    if (
        idx01 >= len(nib01)
        or idx2 >= len(nib2)
        or idx3 >= len(nib3)
        or idx4 >= len(nib4)
    ):
        raise RuntimeError(f"Nibble index out of range: {patinfo}")
    # KR: nib01(8개) + nib2(4개) + nib3(4개) + nib4(4개) = 20비트 패턴
    # EN: nib01(8) + nib2(4) + nib3(4) + nib4(4) = 20-bit pattern
    return list(nib01[idx01]) + list(nib2[idx2]) + list(nib3[idx3]) + list(nib4[idx4])


@lru_cache(maxsize=2048)
def _ps5_build_bc_lut_cached(
    block_w: int,
    block_h: int,
    bytes_per_block: int,
    mode_name: str,
    pipe_log2: int,
    pipe_bank_xor: int,
) -> tuple[int, ...] | None:
    """KR: 캐시된 BC 스위즐 LUT 구축. 패턴 헤더 파일이 있으면 사용하고, 없으면 레이아웃 규칙으로 폴백한다.
    EN: Build a cached BC swizzle LUT. Uses pattern header file if available, falls back to layout rules otherwise.
    """
    tables = _ps5_load_bc_pattern_tables()
    if tables is None:
        # KR: 패턴 테이블 없음: 레이아웃 규칙 기반 폴백
        # EN: No pattern tables: fallback to layout-rule based approach
        return _ps5_build_bc_lut_from_layout_rules(
            block_w,
            block_h,
            bytes_per_block,
            mode_name,
            pipe_bank_xor,
        )
    mode_info = _PS5_BC_MODE_INFO.get(mode_name)
    if mode_info is None:
        return None
    _, _, block_bits, is_xor_mode = mode_info
    patinfo_rows = tables["patinfo_tables"].get(mode_name, [])
    pat_index = int(
        math.log2(bytes_per_block)
    )  # KR: 블록 크기의 log2를 패턴 인덱스로 사용 / EN: Use log2 of block size as pattern index
    if pat_index < 0 or pat_index >= len(patinfo_rows):
        # KR: 해당 블록 크기에 대한 패턴 정보가 없으면 레이아웃 규칙으로 폴백
        # EN: No pattern info for this block size; fallback to layout rules
        return _ps5_build_bc_lut_from_layout_rules(
            block_w,
            block_h,
            bytes_per_block,
            mode_name,
            pipe_bank_xor,
        )
    # KR: 니블 테이블 4개를 결합하여 완전한 패턴 비트 배열 생성
    # EN: Combine 4 nibble tables into a complete pattern bit array
    pattern_bits = _ps5_build_full_pattern(
        tables["nib01"],
        tables["nib2"],
        tables["nib3"],
        tables["nib4"],
        patinfo_rows[pat_index],
    )

    total = block_w * block_h
    lut: list[int] = [0] * total

    # KR: thin 블록 차원 계산 및 피치 정렬
    # EN: Compute thin block dimensions and pitch alignment
    blk_w, blk_h = _ps5_compute_thin_block_dim(block_bits, bytes_per_block)
    pitch_aligned = ((block_w + blk_w - 1) // blk_w) * blk_w
    pitch_blocks = pitch_aligned // blk_w

    # KR: pipe/bank XOR 오프셋 계산을 위한 비트 마스크 준비
    # EN: Prepare bit masks for pipe/bank XOR offset calculation
    blk_mask = (1 << block_bits) - 1
    pipe_interleave_log2 = 8  # KR: Addrlib 파이프 인터리브 기본값: 256바이트 / EN: Addrlib pipe interleave default: 256 bytes
    column_bits = 2
    bank_bits_cap = 4
    bank_xor_bits = max(
        0,
        min(
            block_bits - pipe_interleave_log2 - pipe_log2 - column_bits,
            bank_bits_cap,
        ),
    )
    pipe_mask = (1 << pipe_log2) - 1 if pipe_log2 > 0 else 0
    bank_mask = (
        ((1 << bank_xor_bits) - 1) << (pipe_log2 + column_bits)
        if bank_xor_bits > 0
        else 0
    )
    # KR: XOR 모드인 경우 pipe_bank_xor를 블록 내 오프셋으로 변환
    # EN: For XOR mode, convert pipe_bank_xor to in-block offset
    pb_xor_off = 0
    if is_xor_mode:
        pb_xor_off = (
            (pipe_bank_xor & (pipe_mask | bank_mask)) << pipe_interleave_log2
        ) & blk_mask

    elem_log2 = int(
        math.log2(bytes_per_block)
    )  # KR: 요소 크기의 log2 / EN: log2 of element size
    for y in range(block_h):
        yb = y // blk_h  # KR: 블록 행 인덱스 / EN: Block row index
        row_base = y * block_w
        for x in range(block_w):
            xb = x // blk_w  # KR: 블록 열 인덱스 / EN: Block column index
            blk_idx = (
                yb * pitch_blocks + xb
            )  # KR: 선형 블록 인덱스 / EN: Linear block index
            # KR: 패턴 비트에서 블록 내 오프셋 계산
            # EN: Compute in-block offset from pattern bits
            blk_off = _ps5_compute_offset(pattern_bits, block_bits, x, y, 0, 0)
            # KR: 블록 인덱스와 XOR 오프셋을 결합하여 최종 주소 생성
            # EN: Combine block index and XOR offset to generate final address
            addr = (blk_idx << block_bits) + (blk_off ^ pb_xor_off)
            swizzled_idx = (
                addr >> elem_log2
            )  # KR: 바이트 주소를 요소 인덱스로 변환 / EN: Convert byte address to element index
            linear_idx = row_base + x
            lut[linear_idx] = swizzled_idx % total

    return tuple(lut)


def _ps5_unswizzle_bc_blocks(
    raw: bytes,
    block_w: int,
    block_h: int,
    bytes_per_block: int,
    lut: tuple[int, ...],
) -> bytes:
    """KR: LUT를 사용하여 스위즐된 BC 블록 데이터를 선형 순서로 역스위즐한다.
    EN: Unswizzle swizzled BC block data to linear order using a LUT.
    """
    total = block_w * block_h
    src = memoryview(raw[: total * bytes_per_block])
    dst = bytearray(total * bytes_per_block)
    # KR: LUT의 각 항목: linear_idx -> swizzled_idx 매핑
    # EN: Each LUT entry: linear_idx -> swizzled_idx mapping
    for linear_idx, swizzled_idx in enumerate(lut):
        src_off = (
            swizzled_idx * bytes_per_block
        )  # KR: 스위즐된 소스 오프셋 / EN: Swizzled source offset
        dst_off = (
            linear_idx * bytes_per_block
        )  # KR: 선형 대상 오프셋 / EN: Linear destination offset
        dst[dst_off : dst_off + bytes_per_block] = src[
            src_off : src_off + bytes_per_block
        ]
    return bytes(dst)


def _ps5_decode_bc_to_rgba(
    raw_bytes: bytes,
    pixel_width: int,
    pixel_height: int,
    texture_format: int,
) -> bytes | None:
    """KR: BC 압축 텍스처 데이터를 RGBA 픽셀 데이터로 디코딩한다.
    EN: Decode BC-compressed texture data to RGBA pixel data.
    """
    if texture2ddecoder is None:
        return None
    bc_info = _PS5_BC_FORMATS.get(texture_format)
    if bc_info is None:
        return None
    _, _, _, decoder_name = bc_info
    # KR: 텍스처 포맷에 해당하는 디코더 함수 조회
    # EN: Look up decoder function for the texture format
    decoder = getattr(texture2ddecoder, decoder_name, None)
    if not callable(decoder):
        return None
    try:
        return bytes(decoder(raw_bytes, pixel_width, pixel_height))
    except Exception:
        return None


def _ps5_swap_rb_image(image: Image.Image) -> Image.Image:
    """KR: 이미지의 R(적색)과 B(청색) 채널을 교환한다 (BGR <-> RGB 변환).
    EN: Swap R (red) and B (blue) channels of an image (BGR <-> RGB conversion).
    """
    rgba = image.convert("RGBA")
    r, g, b, a = rgba.split()
    return Image.merge("RGBA", (b, g, r, a))


def _ps5_should_swap_rb_for_bc_preview(texture_format: int) -> bool:
    """KR: PS5 BC 텍스처 프리뷰 시 R/B 채널 교환이 필요한지 판별한다.
    EN: Determine whether R/B channel swap is needed for PS5 BC texture preview.

    PS5 BC 표면은 이 경로에서 BGR 성분 순서로 디코딩되므로 R/B 교환이 필요하다.
    """
    return int(texture_format) in _PS5_BC_FORMATS


def _ps5_crop_blocks_top_left(
    block_data: bytes,
    physical_block_w: int,
    logical_block_w: int,
    logical_block_h: int,
    bytes_per_block: int,
) -> bytes:
    """KR: 물리 블록 그리드에서 논리 블록 영역(좌상단)만 잘라낸다.
    EN: Crop only the logical block region (top-left) from the physical block grid.

    BC 블록 압축 시 물리 그리드는 타일 정렬 요건에 따라 논리 크기보다 클 수 있다.
    각 행에서 논리 너비만큼만 복사하여 패딩 블록을 제거한다.
    """
    if (
        physical_block_w <= 0
        or logical_block_w <= 0
        or logical_block_h <= 0
        or bytes_per_block <= 0
    ):
        return block_data
    logical_size = logical_block_w * logical_block_h * bytes_per_block
    if physical_block_w == logical_block_w:
        return block_data[:logical_size]
    src = memoryview(block_data)
    out = bytearray(logical_size)
    for y in range(logical_block_h):
        src_off = (y * physical_block_w) * bytes_per_block
        dst_off = (y * logical_block_w) * bytes_per_block
        row_bytes = logical_block_w * bytes_per_block
        out[dst_off : dst_off + row_bytes] = src[src_off : src_off + row_bytes]
    return bytes(out)


def _ps5_unswizzle_addrlib_uncompressed_candidate(
    data: bytes,
    width: int,
    height: int,
    bytes_per_element: int,
) -> tuple[bytes, float] | None:
    """KR: 비압축 텍스처에 대해 Addrlib 4KB_S 역스위즐을 시도한다.
    EN: Attempt Addrlib 4KB_S unswizzle for uncompressed textures.

    물리 그리드 크기를 추론하여 LUT 기반 역스위즐 후 논리 영역만 잘라 반환한다.
    """
    if bytes_per_element not in {2, 4}:
        return None
    total_elements = width * height
    if total_elements <= 0 or total_elements > 2_000_000:
        return None

    logical_bytes = total_elements * bytes_per_element
    usable = data[: (len(data) // bytes_per_element) * bytes_per_element]
    if len(usable) < logical_bytes:
        return None

    physical_total = len(usable) // bytes_per_element
    inferred_w, inferred_h = _ps5_infer_physical_grid(
        physical_total,
        width,
        height,
        align_width=8,
        align_height=8,
    )
    candidates: list[tuple[int, int]] = [(width, height)]
    if (
        inferred_w * inferred_h == physical_total
        and (inferred_w, inferred_h) not in candidates
    ):
        candidates.append((inferred_w, inferred_h))

    for physical_w, physical_h in candidates:
        physical_bytes = physical_w * physical_h * bytes_per_element
        if physical_bytes > len(usable):
            continue
        lut = _ps5_build_bc_lut_cached(
            physical_w,
            physical_h,
            bytes_per_element,
            "4KB_S",
            2,
            0,
        )
        if lut is None:
            continue
        unsw_full = _ps5_unswizzle_bc_blocks(
            usable[:physical_bytes],
            physical_w,
            physical_h,
            bytes_per_element,
            lut,
        )
        unsw_logical = _ps5_crop_blocks_top_left(
            unsw_full,
            physical_w,
            width,
            height,
            bytes_per_element,
        )
        return unsw_logical, 0.0

    return None


def _ps5_pipe_bank_xor_span(
    mode_name: str, bytes_per_block: int, pipe_log2: int
) -> int:
    """KR: 주어진 모드의 pipe/bank XOR 값 범위를 계산한다.
    EN: Compute the pipe/bank XOR value range for the given mode.

    XOR 모드가 아니면 1을 반환한다. XOR 모드인 경우 pipe 비트와 bank 비트의
    합으로 가능한 조합 수(2^total_bits)를 계산하여 반환한다.
    """
    mode_info = _PS5_BC_MODE_INFO.get(mode_name)
    if mode_info is None:
        return 0
    _, _, block_bits, is_xor_mode = mode_info
    if not is_xor_mode:
        return 1
    pipe_log2 = max(0, int(pipe_log2))
    pipe_mask_bits = pipe_log2
    bank_xor_bits = max(
        0,
        min(
            block_bits - 8 - pipe_log2 - 2,
            4,
        ),
    )
    total_bits = pipe_mask_bits + bank_xor_bits
    if total_bits <= 0:
        return 1
    return 1 << total_bits


def _ps5_iter_pipe_bank_xor_values(
    mode_name: str,
    bytes_per_block: int,
    pipe_log2: int,
    *,
    exhaustive: bool = False,
) -> tuple[int, ...]:
    """KR: 역스위즐 후보 탐색용 pipe/bank XOR 값 목록을 생성한다.
    EN: Generate a list of pipe/bank XOR values for unswizzle candidate search.

    exhaustive=True이면 전체 범위를 반환하고, 아니면 빠른 탐색용 축약 목록을 반환한다.
    """
    span = _ps5_pipe_bank_xor_span(mode_name, bytes_per_block, pipe_log2)
    if span <= 1:
        return (0,)
    if exhaustive:
        return tuple(range(span))
    # KR: 기본 경로는 빠르게 처리하고, 필요 시에만 전수 탐색으로 전환
    # EN: Default path handles quickly; switch to exhaustive search only when needed
    quick = tuple(v for v in (0, 1, 2, 3, 4, 7) if v < span)
    return quick if quick else (0,)


def _ps5_unswizzle_bc_best_candidate(
    raw: bytes,
    pixel_width: int,
    pixel_height: int,
    texture_format: int,
    *,
    mode_candidates: Iterable[str] | None = None,
    pipe_log2_candidates: Iterable[int] | None = None,
    exhaustive: bool = False,
    exhaustive_xor: bool = False,
) -> tuple[bytes, str | None, float | None, tuple[int, int], tuple[int, int]] | None:
    """KR: BC 블록 압축 텍스처의 최적 역스위즐 후보를 점수 기반으로 선택한다.
    EN: Select the best unswizzle candidate for BC block-compressed textures based on scoring.

    모드/pipe/XOR 조합별로 역스위즐한 뒤 RGBA 디코딩하여 거칠기 점수를 비교한다.
    가장 낮은 거칠기 점수를 가진 후보의 블록 데이터와 모드 정보를 반환한다.
    """
    bc_info = _PS5_BC_FORMATS.get(texture_format)
    if bc_info is None:
        return None
    block_w_px, block_h_px, bytes_per_block, _ = bc_info
    logical_block_w = (pixel_width + block_w_px - 1) // block_w_px
    logical_block_h = (pixel_height + block_h_px - 1) // block_h_px
    logical_block_total = logical_block_w * logical_block_h
    logical_bytes = logical_block_total * bytes_per_block

    usable = raw[: (len(raw) // bytes_per_block) * bytes_per_block]
    if len(usable) < logical_bytes:
        return None

    physical_total_blocks = len(usable) // bytes_per_block
    align = 16 if bytes_per_block >= 16 else 8

    raw_logical = usable[:logical_bytes]
    raw_rgba = _ps5_decode_bc_to_rgba(
        raw_logical, pixel_width, pixel_height, texture_format
    )
    raw_score = (
        _ps5_roughness_score(raw_rgba, pixel_width, pixel_height, 4)
        if raw_rgba is not None
        else None
    )

    modes = (
        list(mode_candidates)
        if mode_candidates is not None
        else (
            list(_PS5_BC_MODE_INFO.keys())
            if exhaustive
            else list(_PS5_BC_FAST_MODE_NAMES)
        )
    )
    pipe_candidates = (
        tuple(pipe_log2_candidates)
        if pipe_log2_candidates is not None
        else ((0, 1, 2, 3) if exhaustive else (2, 1, 3))
    )

    best_raw = raw_logical
    best_mode: str | None = None
    best_ratio: float | None = None
    best_score: float | None = None

    best_physical = (logical_block_w, logical_block_h)

    for mode_name in modes:
        physical_candidates = _ps5_physical_grid_candidates_for_mode(
            physical_total_blocks,
            logical_block_w,
            logical_block_h,
            bytes_per_block=bytes_per_block,
            mode_name=mode_name,
            align_width=align,
            align_height=align,
        )
        for physical_block_w, physical_block_h in physical_candidates:
            physical_bytes = physical_block_w * physical_block_h * bytes_per_block
            if physical_bytes > len(usable):
                continue
            source_for_layout = usable[:physical_bytes]

            for pipe_log2 in pipe_candidates:
                pipe_bank_xor_values = _ps5_iter_pipe_bank_xor_values(
                    mode_name,
                    bytes_per_block,
                    pipe_log2,
                    exhaustive=exhaustive_xor,
                )
                for pipe_bank_xor in pipe_bank_xor_values:
                    lut = _ps5_build_bc_lut_cached(
                        physical_block_w,
                        physical_block_h,
                        bytes_per_block,
                        mode_name,
                        pipe_log2,
                        pipe_bank_xor,
                    )
                    if lut is None:
                        continue
                    unsw_full = _ps5_unswizzle_bc_blocks(
                        source_for_layout,
                        physical_block_w,
                        physical_block_h,
                        bytes_per_block,
                        lut,
                    )
                    unsw_logical = _ps5_crop_blocks_top_left(
                        unsw_full,
                        physical_block_w,
                        logical_block_w,
                        logical_block_h,
                        bytes_per_block,
                    )

                    if raw_score is None:
                        if best_mode is None:
                            best_raw = unsw_logical
                            best_mode = f"{mode_name}:p{pipe_log2}:x{pipe_bank_xor}"
                            best_physical = (physical_block_w, physical_block_h)
                        continue

                    rgba = _ps5_decode_bc_to_rgba(
                        unsw_logical, pixel_width, pixel_height, texture_format
                    )
                    if rgba is None:
                        continue
                    score = _ps5_roughness_score(rgba, pixel_width, pixel_height, 4)
                    ratio = (score / raw_score) if raw_score > 0 else None
                    if best_score is None or score < best_score:
                        best_score = score
                        best_ratio = ratio
                        best_mode = f"{mode_name}:p{pipe_log2}:x{pipe_bank_xor}"
                        best_raw = unsw_logical
                        best_physical = (physical_block_w, physical_block_h)

    return (
        best_raw,
        best_mode,
        best_ratio,
        (logical_block_w, logical_block_h),
        best_physical,
    )


def _ps5_try_end_aligned_4kb_s_candidate(
    usable: bytes,
    logical_block_w: int,
    logical_block_h: int,
    bytes_per_block: int,
) -> tuple[bytes, str, tuple[int, int]] | None:
    """KR: 스트림 끝 정렬 기반 4KB_S 역스위즐 후보를 시도한다.
    EN: Try a stream end-aligned 4KB_S unswizzle candidate.

    mip 레벨이 여러 개인 경우 mip0 데이터가 스트림 끝에 위치할 수 있으므로,
    끝 기준 오프셋에서 물리 블록 크기만큼 잘라 4KB_S LUT로 역스위즐한다.
    """
    bits = _ps5_tile_bit_dimensions_for_mode("4KB_S", bytes_per_block)
    if bits is None:
        return None
    tile_w = 1 << bits[0]
    tile_h = 1 << bits[1]
    if tile_w <= 0 or tile_h <= 0:
        return None

    physical_block_w = _ps5_align_up(logical_block_w, tile_w)
    physical_block_h = _ps5_align_up(logical_block_h, tile_h)
    physical_bytes = physical_block_w * physical_block_h * bytes_per_block
    if physical_bytes <= 0 or physical_bytes > len(usable):
        return None

    offset_bytes = len(usable) - physical_bytes
    source_for_layout = usable[offset_bytes : offset_bytes + physical_bytes]
    lut = _ps5_build_bc_lut_cached(
        physical_block_w,
        physical_block_h,
        bytes_per_block,
        "4KB_S",
        2,
        0,
    )
    if lut is None:
        return None
    unsw_full = _ps5_unswizzle_bc_blocks(
        source_for_layout,
        physical_block_w,
        physical_block_h,
        bytes_per_block,
        lut,
    )
    unsw_logical = _ps5_crop_blocks_top_left(
        unsw_full,
        physical_block_w,
        logical_block_w,
        logical_block_h,
        bytes_per_block,
    )
    return (
        unsw_logical,
        f"4KB_S:p2:x0:o{offset_bytes}",
        (physical_block_w, physical_block_h),
    )


def _ps5_unswizzle_bc_best_layout_match(
    raw: bytes,
    pixel_width: int,
    pixel_height: int,
    texture_format: int,
    *,
    mip_count: int | None = None,
) -> tuple[bytes, str | None, float | None, tuple[int, int], tuple[int, int]] | None:
    """KR: 고정 우선순위로 첫 번째 유효한 BC 레이아웃 변형을 선택한다.
    EN: Select the first valid BC layout variant using fixed priority.

    mip 레벨 오프셋을 계산하여 mip0 시작 위치를 결정한 뒤, 모드/pipe/XOR
    조합을 순회하며 첫 번째 유효한 역스위즐 결과를 반환한다.
    비정방 텍스처의 경우 end-aligned 4KB_S 후보도 추가로 확인한다.
    """
    bc_info = _PS5_BC_FORMATS.get(texture_format)
    if bc_info is None:
        return None
    block_w_px, block_h_px, bytes_per_block, _ = bc_info
    logical_block_w = (pixel_width + block_w_px - 1) // block_w_px
    logical_block_h = (pixel_height + block_h_px - 1) // block_h_px
    logical_block_total = logical_block_w * logical_block_h
    logical_bytes = logical_block_total * bytes_per_block

    usable = raw[: (len(raw) // bytes_per_block) * bytes_per_block]
    if len(usable) < logical_bytes:
        return None
    source_window = usable
    mip0_offset_bytes = 0
    if mip_count is not None and int(mip_count) > 1:
        # KR: 높은 mip부터 offset이 누적되므로 mip0가 tail 뒤에서 시작할 수 있다
        # EN: Offset accumulates from higher mips, so mip0 may start after the tail
        lower_tail_sum = 0
        w = max(1, int(pixel_width))
        h = max(1, int(pixel_height))
        levels: list[int] = []
        level_count = max(1, int(mip_count))
        for _ in range(level_count):
            bw = max(1, (w + block_w_px - 1) // block_w_px)
            bh = max(1, (h + block_h_px - 1) // block_h_px)
            levels.append(bw * bh * bytes_per_block)
            w = max(1, w >> 1)
            h = max(1, h >> 1)
        if len(levels) > 1:
            # KR: tail packing은 mip별 256B, tail 2KB 정렬을 사용한다
            # EN: Tail packing uses 256B per mip, 2KB tail alignment
            for level_bytes in levels[1:]:
                lower_tail_sum += _ps5_align_up(level_bytes, 0x100)
            mip0_offset_bytes = _ps5_align_up(lower_tail_sum, 0x800)
            base_alloc = _ps5_align_up(levels[0], 0x800)
            modeled_total = mip0_offset_bytes + base_alloc
            if modeled_total < len(usable):
                # KR: 비정방 케이스에서는 stream 끝 기준으로 mip0를 다시 맞춘다
                # EN: For non-square cases, re-align mip0 from stream end
                mip0_offset_bytes += len(usable) - modeled_total
            if mip0_offset_bytes + base_alloc <= len(usable):
                base_end = mip0_offset_bytes + base_alloc
                source_window = usable[mip0_offset_bytes:base_end]
            elif mip0_offset_bytes + levels[0] <= len(usable):
                base_end = mip0_offset_bytes + levels[0]
                source_window = usable[mip0_offset_bytes:base_end]
            else:
                mip0_offset_bytes = 0
                source_window = usable

    if len(source_window) < logical_bytes:
        return None
    raw_logical = source_window[:logical_bytes]

    physical_total_blocks = len(source_window) // bytes_per_block
    align = 16 if bytes_per_block >= 16 else 8

    # KR: 4KB_S 모드를 최우선으로 시도한다
    # EN: Try 4KB_S mode with highest priority
    mode_order: list[str] = ["4KB_S"]
    for mode_name in _PS5_BC_FAST_MODE_NAMES:
        if mode_name not in mode_order:
            mode_order.append(mode_name)
    for mode_name in _PS5_BC_MODE_INFO:
        if mode_name not in mode_order:
            mode_order.append(mode_name)
    pipe_order = (2, 1, 3, 0)

    for mode_name in mode_order:
        physical_candidates = _ps5_physical_grid_candidates_for_mode(
            physical_total_blocks,
            logical_block_w,
            logical_block_h,
            bytes_per_block=bytes_per_block,
            mode_name=mode_name,
            align_width=align,
            align_height=align,
        )
        for physical_block_w, physical_block_h in physical_candidates:
            physical_bytes = physical_block_w * physical_block_h * bytes_per_block
            if physical_bytes > len(source_window):
                continue
            source_for_layout = source_window[:physical_bytes]
            for pipe_log2 in pipe_order:
                for pipe_bank_xor in _ps5_iter_pipe_bank_xor_values(
                    mode_name,
                    bytes_per_block,
                    pipe_log2,
                    exhaustive=True,
                ):
                    lut = _ps5_build_bc_lut_cached(
                        physical_block_w,
                        physical_block_h,
                        bytes_per_block,
                        mode_name,
                        pipe_log2,
                        pipe_bank_xor,
                    )
                    if lut is None:
                        continue
                    unsw_full = _ps5_unswizzle_bc_blocks(
                        source_for_layout,
                        physical_block_w,
                        physical_block_h,
                        bytes_per_block,
                        lut,
                    )
                    unsw_logical = _ps5_crop_blocks_top_left(
                        unsw_full,
                        physical_block_w,
                        logical_block_w,
                        logical_block_h,
                        bytes_per_block,
                    )
                    if (
                        mip_count is not None
                        and int(mip_count) > 1
                        and mode_name.startswith("256B_")
                    ):
                        # KR: 일부 비정방 케이스는 end-aligned 4KB_S 레이아웃으로 재확인한다
                        # EN: Some non-square cases need re-verification with end-aligned 4KB_S layout
                        alt = _ps5_try_end_aligned_4kb_s_candidate(
                            usable,
                            logical_block_w,
                            logical_block_h,
                            bytes_per_block,
                        )
                        if alt is not None:
                            alt_raw, alt_mode, alt_physical = alt
                            return (
                                alt_raw,
                                alt_mode,
                                None,
                                (logical_block_w, logical_block_h),
                                alt_physical,
                            )
                    return (
                        unsw_logical,
                        (
                            f"{mode_name}:p{pipe_log2}:x{pipe_bank_xor}:o{mip0_offset_bytes}"
                            if mip0_offset_bytes > 0
                            else f"{mode_name}:p{pipe_log2}:x{pipe_bank_xor}"
                        ),
                        None,
                        (logical_block_w, logical_block_h),
                        (physical_block_w, physical_block_h),
                    )

    return (
        raw_logical,
        None,
        None,
        (logical_block_w, logical_block_h),
        (logical_block_w, logical_block_h),
    )


@lru_cache(maxsize=128)
def _ps5_bit_positions(mask: int) -> tuple[int, ...]:
    """KR: 마스크에서 세트된 비트 위치 목록을 반환합니다.
    EN: Return a list of set bit positions in a mask.
    """
    return tuple(i for i in range(max(mask.bit_length(), 0)) if (mask >> i) & 1)


@lru_cache(maxsize=128)
def _ps5_axis_tile_size(mask: int) -> int:
    """KR: 마스크의 세트된 비트 수로부터 축별 타일 크기를 계산합니다.
    EN: Compute axis tile size from the number of set bits in a mask.
    """
    positions = _ps5_bit_positions(mask)
    return 1 << len(positions) if positions else 1


@lru_cache(maxsize=128)
def _ps5_deposit_table(mask: int) -> tuple[int, ...]:
    """KR: 마스크 비트폭(타일 기준) pdep 유사 배치 테이블을 생성합니다.
    EN: Generate a pdep-like deposit table for the mask bit width (tile-based).
    """
    positions = _ps5_bit_positions(mask)
    axis_size = _ps5_axis_tile_size(mask)
    table: list[int] = [0] * axis_size
    for value in range(axis_size):
        deposited = 0
        for bit_index, dst_bit in enumerate(positions):
            if (value >> bit_index) & 1:
                deposited |= 1 << dst_bit
        table[value] = deposited
    return tuple(table)


def _ps5_validate_texture_shape(
    data: bytes, width: int, height: int, bytes_per_element: int
) -> int:
    """KR: 텍스처 크기/데이터 길이를 검증하고 총 요소 수를 반환합니다.
    EN: Validate texture dimensions/data length and return the total element count.
    """
    if width <= 0 or height <= 0 or bytes_per_element <= 0:
        raise ValueError(
            f"Invalid texture shape for swizzle: width={width}, height={height}, bpe={bytes_per_element}"
        )
    total_elements = width * height
    expected_size = total_elements * bytes_per_element
    if len(data) < expected_size:
        raise ValueError(
            f"Texture data size mismatch: expected_at_least={expected_size}, got={len(data)} "
            f"(w={width}, h={height}, bpe={bytes_per_element})"
        )
    return total_elements


def _ps5_clip_to_base_level(
    data: bytes,
    width: int,
    height: int,
    bytes_per_element: int,
) -> tuple[bytes, int]:
    """KR: mip0 기본 레벨만 남기고 초과 바이트를 잘라냅니다.
    EN: Clip to mip0 base level only, trimming excess bytes.
    """
    total_elements = _ps5_validate_texture_shape(data, width, height, bytes_per_element)
    expected_size = total_elements * bytes_per_element
    if len(data) > expected_size:
        return data[:expected_size], len(data) - expected_size
    return data, 0


def _texture_format_enum_name(texture_format: int) -> str:
    """KR: 텍스처 포맷 정수를 UnityPy enum 이름 문자열로 변환합니다.
    EN: Convert a texture format integer to a UnityPy enum name string.
    """
    value = int(texture_format)
    if _UnityTextureFormatEnum is not None:
        try:
            return str(_UnityTextureFormatEnum(value).name)
        except Exception:
            pass
    return f"TextureFormat_{value}"


def _texture_format_layout_details(texture_format: int) -> dict[str, Any] | None:
    """KR: 텍스처 포맷의 PS5 레이아웃 메타데이터(블록 크기, 디코더 등)를 반환합니다.
    EN: Return PS5 layout metadata (block size, decoder, etc.) for a texture format.
    """
    value = int(texture_format)
    meta = _PS5_LAYOUT_FORMAT_META.get(value)
    if meta is None:
        return None
    block_pack = int(meta["block_pack"])
    bytes_per_block, block_w, block_h, depth = _ps5_unpack_block_pack(block_pack)
    flags_word = _PS5_LAYOUT_FORMAT_FLAGS.get(value)
    layout_shift = ((flags_word & 0x6) * 2 + 8) if flags_word is not None else None
    mode_4kb_s_triplet = _PS5_4KB_S_TRIPLETS_BY_BLOCK_BYTES.get(bytes_per_block)
    return {
        "label": str(meta["label"]),
        "word0": int(meta["word0"]),
        "block_pack": block_pack,
        "bytes_per_block": bytes_per_block,
        "block_width": block_w,
        "block_height": block_h,
        "block_depth": depth,
        "decoder": _PS5_BC_DECODER_BY_FORMAT.get(value),
        "flags_word": int(flags_word) if flags_word is not None else None,
        "layout_shift": int(layout_shift) if layout_shift is not None else None,
        "mode_4kb_s_triplet": list(mode_4kb_s_triplet)
        if mode_4kb_s_triplet is not None
        else None,
    }


def _texture_format_bytes_per_element(texture_format: int) -> int | None:
    """KR: 텍스처 포맷에서 픽셀당 바이트 수(BPE)를 반환합니다.
    EN: Return bytes per pixel (BPE) for a texture format.
    """
    # KR: 가능한 경우 UnityPy enum 이름 기준으로 BPE를 해석 (버전별 숫자 변동 방지)
    # EN: Interpret BPE by UnityPy enum name when possible (prevents version-specific number changes)
    bpe_by_name = {
        "Alpha8": 1,
        "ARGB4444": 2,
        "RGB24": 3,
        "RGBA32": 4,
        "ARGB32": 4,
        "RGB565": 2,
        "R16": 2,
        "RG16": 2,
        "R8": 1,
    }
    value: int | None = None
    enum_name = _texture_format_enum_name(texture_format)
    if enum_name.startswith("TextureFormat_"):
        enum_name = ""
    if enum_name:
        value = bpe_by_name.get(enum_name)

    # KR: enum 해석 실패 시 최소 숫자 fallback.
    # EN: Minimal numeric fallback when enum interpretation fails.
    if value is None:
        format_to_bpe = {
            1: 1,  # Alpha8
            2: 2,  # ARGB4444
            3: 3,  # RGB24
            4: 4,  # RGBA32
            5: 4,  # ARGB32
            7: 2,  # RGB565
            9: 2,  # R16
            62: 2,  # RG16
            63: 1,  # R8
        }
        value = format_to_bpe.get(int(texture_format), None)

    if value in {1, 2, 3, 4}:
        return value
    return None


def _texture_format_is_bc(texture_format: int) -> bool:
    """KR: BC(블록 압축) 포맷 여부를 반환합니다.
    EN: Return whether the format is BC (block compressed).
    """
    return int(texture_format) in _PS5_BC_FORMATS


def _texture_format_is_crunched(texture_format: int) -> bool:
    """KR: Crunched(크런치 압축) 포맷 여부를 반환합니다.
    EN: Return whether the format is Crunched (crunch compressed).
    """
    value = int(texture_format)
    if value in {28, 29}:  # DXT1Crunched / DXT5Crunched
        return True
    enum_name = _texture_format_enum_name(value)
    return enum_name in {
        "DXT1Crunched",
        "DXT5Crunched",
        "ETC_RGB4Crunched",
        "ETC2_RGBA8Crunched",
    }


def ps5_unswizzle_bytes(
    data: bytes,
    width: int,
    height: int,
    bytes_per_element: int,
    mask_x: int | None = None,
    mask_y: int | None = None,
) -> bytes:
    """KR: PS5 swizzled 바이트 배열을 선형(행 우선) 순서로 변환합니다.
    mask_x/mask_y가 None이면 width/height에서 자동 계산합니다.
    EN: Convert a PS5 swizzled byte array to linear (row-major) order.
    Automatically computes mask_x/mask_y from width/height if None.
    """
    if not _ps5_dimensions_supported(width, height, bytes_per_element):
        clipped, _ = _ps5_clip_to_base_level(data, width, height, bytes_per_element)
        return clipped
    if mask_x is None or mask_y is None:
        mask_x, mask_y = compute_ps5_swizzle_masks(width, height, bytes_per_element)
    data, _ = _ps5_clip_to_base_level(data, width, height, bytes_per_element)
    total_elements = _ps5_validate_texture_shape(data, width, height, bytes_per_element)
    src = memoryview(data)
    dst = bytearray(len(data))
    tile_w = _ps5_axis_tile_size(mask_x)
    tile_h = _ps5_axis_tile_size(mask_y)
    xdep = _ps5_deposit_table(mask_x)
    ydep = _ps5_deposit_table(mask_y)
    macro_cols = (width + tile_w - 1) // tile_w
    tile_elements = tile_w * tile_h

    for y in range(height):
        row_start = y * width
        macro_y = y // tile_h
        local_y = y % tile_h
        row_offset = ydep[local_y]
        for x in range(width):
            macro_x = x // tile_w
            local_x = x % tile_w
            tile_base = ((macro_y * macro_cols) + macro_x) * tile_elements
            src_idx = tile_base + row_offset + xdep[local_x]
            if src_idx < 0 or src_idx >= total_elements:
                raise ValueError(
                    f"PS5 unswizzle index out of range: idx={src_idx}, total={total_elements}, "
                    f"w={width}, h={height}, mask_x={mask_x:#x}, mask_y={mask_y:#x}"
                )
            src_off = src_idx * bytes_per_element
            dst_off = (row_start + x) * bytes_per_element
            dst[dst_off : dst_off + bytes_per_element] = src[
                src_off : src_off + bytes_per_element
            ]

    return bytes(dst)


def ps5_swizzle_bytes(
    data: bytes,
    width: int,
    height: int,
    bytes_per_element: int,
    mask_x: int | None = None,
    mask_y: int | None = None,
) -> bytes:
    """KR: 선형(행 우선) 바이트 배열을 PS5 swizzle 순서로 변환합니다.
    mask_x/mask_y가 None이면 width/height에서 자동 계산합니다.
    EN: Convert a linear (row-major) byte array to PS5 swizzle order.
    Automatically computes mask_x/mask_y from width/height if None.
    """
    if not _ps5_dimensions_supported(width, height, bytes_per_element):
        clipped, _ = _ps5_clip_to_base_level(data, width, height, bytes_per_element)
        return clipped
    if mask_x is None or mask_y is None:
        mask_x, mask_y = compute_ps5_swizzle_masks(width, height, bytes_per_element)
    data, _ = _ps5_clip_to_base_level(data, width, height, bytes_per_element)
    total_elements = _ps5_validate_texture_shape(data, width, height, bytes_per_element)
    src = memoryview(data)
    dst = bytearray(len(data))
    tile_w = _ps5_axis_tile_size(mask_x)
    tile_h = _ps5_axis_tile_size(mask_y)
    xdep = _ps5_deposit_table(mask_x)
    ydep = _ps5_deposit_table(mask_y)
    macro_cols = (width + tile_w - 1) // tile_w
    tile_elements = tile_w * tile_h

    for y in range(height):
        row_start = y * width
        macro_y = y // tile_h
        local_y = y % tile_h
        row_offset = ydep[local_y]
        for x in range(width):
            macro_x = x // tile_w
            local_x = x % tile_w
            tile_base = ((macro_y * macro_cols) + macro_x) * tile_elements
            dst_idx = tile_base + row_offset + xdep[local_x]
            if dst_idx < 0 or dst_idx >= total_elements:
                raise ValueError(
                    f"PS5 swizzle index out of range: idx={dst_idx}, total={total_elements}, "
                    f"w={width}, h={height}, mask_x={mask_x:#x}, mask_y={mask_y:#x}"
                )
            src_off = (row_start + x) * bytes_per_element
            dst_off = dst_idx * bytes_per_element
            dst[dst_off : dst_off + bytes_per_element] = src[
                src_off : src_off + bytes_per_element
            ]

    return bytes(dst)


def _ps5_mode_for_swizzle(image: Image.Image) -> str:
    """KR: swizzle 처리에 적합한 PIL 모드를 결정합니다.
    EN: Determine the PIL mode suitable for swizzle processing.
    """
    mode = image.mode
    if mode in {"L", "LA", "RGB", "RGBA"}:
        return mode
    if mode == "P":
        return "L"
    return "RGBA"


def _ps5_prepare_image(image: Image.Image) -> Image.Image:
    """KR: swizzle 처리 전 이미지를 적합한 모드로 변환합니다.
    EN: Convert image to a suitable mode before swizzle processing.
    """
    mode = _ps5_mode_for_swizzle(image)
    if image.mode == mode:
        return image
    return image.convert(mode)


def _ps5_roughness_score(
    data: bytes,
    width: int,
    height: int,
    bytes_per_element: int,
) -> float:
    """KR: 로컬 픽셀 변화량 기반 거칠기 점수를 계산합니다.
    낮을수록 부드러움 = linear 가능성 높음; swizzled 데이터는 노이즈처럼 보입니다.
    항상 인접 픽셀(step=1)을 비교하여 swizzle/linear를 정확히 판별합니다.
    성능을 위해 전체 픽셀 대신 행/열 서브셋을 샘플링합니다.
    EN: Compute a roughness score based on local pixel variation.
    Lower means smoother = likely linear; swizzled data looks like noise.
    Always compares adjacent pixels (step=1) to accurately distinguish swizzle/linear.
    Samples row/column subsets instead of all pixels for performance.
    """
    data, _ = _ps5_clip_to_base_level(data, width, height, bytes_per_element)
    _ps5_validate_texture_shape(data, width, height, bytes_per_element)
    view = memoryview(data)
    bpe = bytes_per_element

    # KR: --- 측정할 채널 결정 ---
    # EN: --- Determine channel to measure ---
    max_sample_lines = 256
    channel_index = 0
    if bpe > 1:
        # KR: 분산이 가장 높은(정보량이 가장 많은) 채널을 선택
        # EN: Select the channel with highest variance (most information)
        row_step = max(1, height // max_sample_lines)
        col_step = max(1, width // max_sample_lines)
        sums = [0.0] * bpe
        sums_sq = [0.0] * bpe
        sample_count = 0
        for y in range(0, height, row_step):
            row_base = y * width * bpe
            for x in range(0, width, col_step):
                base = row_base + x * bpe
                sample_count += 1
                for ch in range(bpe):
                    value = float(view[base + ch])
                    sums[ch] += value
                    sums_sq[ch] += value * value
        if sample_count > 0:
            best_var = -1.0
            for ch in range(bpe):
                mean = sums[ch] / sample_count
                variance = (sums_sq[ch] / sample_count) - (mean * mean)
                if variance > best_var:
                    best_var = variance
                    channel_index = ch

    # KR: --- dx(수평) 측정: 행 샘플링, 항상 인접 픽셀 비교 ---
    # EN: --- dx (horizontal) measurement: row sampling, always comparing adjacent pixels ---
    dx_sum = 0.0
    dx_count = 0
    row_step = max(1, height // max_sample_lines)
    if width > 1:
        for y in range(0, height, row_step):
            row_base = y * width * bpe
            for x in range(width - 1):
                left_idx = row_base + x * bpe + channel_index
                right_idx = (
                    left_idx + bpe
                )  # step=1, KR: 항상 인접 / EN: always adjacent
                dx_sum += abs(float(view[right_idx]) - float(view[left_idx]))
                dx_count += 1

    # KR: --- dy(수직) 측정: 열 샘플링, 항상 인접 픽셀 비교 ---
    # EN: --- dy (vertical) measurement: column sampling, always comparing adjacent pixels ---
    dy_sum = 0.0
    dy_count = 0
    col_step = max(1, width // max_sample_lines)
    if height > 1:
        row_stride = width * bpe
        for x in range(0, width, col_step):
            col_base = x * bpe + channel_index
            for y in range(height - 1):
                up_idx = col_base + y * row_stride
                down_idx = (
                    up_idx + row_stride
                )  # step=1, KR: 항상 인접 / EN: always adjacent
                dy_sum += abs(float(view[down_idx]) - float(view[up_idx]))
                dy_count += 1

    dx = dx_sum / dx_count if dx_count else 0.0
    dy = dy_sum / dy_count if dy_count else 0.0
    return float(dx + dy)


def detect_ps5_swizzle_state(
    data: bytes,
    width: int,
    height: int,
    bytes_per_element: int,
    mask_x: int | None = None,
    mask_y: int | None = None,
) -> tuple[str, float, float, float, bytes, bytes]:
    """KR: 입력 바이트가 swizzled인지 휴리스틱으로 판별합니다.
    EN: Heuristically determine whether input bytes are swizzled.
    """
    data, _ = _ps5_clip_to_base_level(data, width, height, bytes_per_element)
    if not _ps5_dimensions_supported(width, height, bytes_per_element):
        raw_score = _ps5_roughness_score(data, width, height, bytes_per_element)
        return "inconclusive", raw_score, raw_score, raw_score, data, data
    if mask_x is None or mask_y is None:
        mask_x, mask_y = compute_ps5_swizzle_masks(width, height, bytes_per_element)
    raw_score = _ps5_roughness_score(data, width, height, bytes_per_element)
    unswizzled = ps5_unswizzle_bytes(
        data, width, height, bytes_per_element, mask_x=mask_x, mask_y=mask_y
    )
    swizzled = ps5_swizzle_bytes(
        data, width, height, bytes_per_element, mask_x=mask_x, mask_y=mask_y
    )
    unsw_score = _ps5_roughness_score(unswizzled, width, height, bytes_per_element)
    swz_score = _ps5_roughness_score(swizzled, width, height, bytes_per_element)

    if unsw_score < raw_score * 0.92 and unsw_score <= swz_score * 0.98:
        verdict = "likely_swizzled_input"
    elif raw_score <= unsw_score * 0.92 and raw_score <= swz_score * 0.92:
        verdict = "likely_linear_input"
    else:
        verdict = "inconclusive"

    return verdict, raw_score, unsw_score, swz_score, unswizzled, swizzled


def _ps5_unswizzle_best_variant(
    data: bytes,
    width: int,
    height: int,
    bytes_per_element: int,
    mask_x: int | None = None,
    mask_y: int | None = None,
    allow_axis_swap: bool = False,
    roughness_guard: bool = False,
) -> tuple[bytes, int, int, str, float]:
    """KR: bpe별 축-전치 규칙에 따라 unswizzle 후보를 선택합니다.
    roughness_guard=True이면, unswizzle 결과가 원본보다 거칠 경우 원본을 반환합니다.
    축 전치는 bpe에 따라 달라집니다:
      bpe=1 (Alpha8): 항상 전치 -> (H,W)로 unswizzle 후 90도 회전으로 복원.
      bpe=4 (RGBA32): 전치 안 함 -> (W,H)로 직접 unswizzle.
    EN: Select an unswizzle candidate according to per-bpe axis-transpose rules.
    If roughness_guard=True, returns original when unswizzle result is rougher.
    Axis transposition varies by bpe:
      bpe=1 (Alpha8): always transpose -> unswizzle as (H,W) then restore via 90-degree rotation.
      bpe=4 (RGBA32): no transpose -> unswizzle directly as (W,H).
    """
    logical_bytes = width * height * bytes_per_element
    usable = data[: (len(data) // bytes_per_element) * bytes_per_element]
    clipped = usable[:logical_bytes]

    # KR: roughness guard를 위해 원본 roughness를 미리 계산합니다.
    # EN: Pre-compute original roughness for roughness guard.
    raw_score = (
        _ps5_roughness_score(clipped, width, height, bytes_per_element)
        if roughness_guard
        else None
    )

    # KR: bpe별 축 전치 규칙 결정.
    # EN: Determine axis transpose rule by bpe.
    should_transpose = _PS5_AXIS_TRANSPOSE.get(bytes_per_element, False)

    if (
        allow_axis_swap
        and should_transpose
        and mask_x is None
        and mask_y is None
        and _ps5_dimensions_supported(height, width, bytes_per_element)
    ):
        # KR: 전치 bpe (예: Alpha8): 종횡비와 무관하게 (H,W)로 unswizzle 후 회전 후보로 취급합니다
        #     (정사각형 텍스처 포함).
        # EN: Transposed bpe (e.g. Alpha8): unswizzle as (H,W) regardless of aspect ratio, treated as rotation candidate
        #     (including square textures).
        try:
            swapped = ps5_unswizzle_bytes(
                clipped,
                height,
                width,
                bytes_per_element,
                mask_x=None,
                mask_y=None,
            )
            best_data = swapped
            best_width = height
            best_height = width
            best_variant = "swapped_axes"
            best_score = _ps5_roughness_score(swapped, height, width, bytes_per_element)
        except Exception:
            # KR: 일반 모드로 폴백
            # EN: Fallback to normal mode
            normal = ps5_unswizzle_bytes(
                clipped,
                width,
                height,
                bytes_per_element,
                mask_x=mask_x,
                mask_y=mask_y,
            )
            best_data = normal
            best_width = width
            best_height = height
            best_variant = "normal"
            best_score = _ps5_roughness_score(normal, width, height, bytes_per_element)
    else:
        # KR: 비전치 bpe (예: RGBA32) 또는 정사각형: (W,H)로 unswizzle합니다.
        # EN: Non-transposed bpe (e.g. RGBA32) or square: unswizzle as (W,H).
        normal = ps5_unswizzle_bytes(
            clipped,
            width,
            height,
            bytes_per_element,
            mask_x=mask_x,
            mask_y=mask_y,
        )
        best_data = normal
        best_width = width
        best_height = height
        best_variant = "normal"
        best_score = _ps5_roughness_score(normal, width, height, bytes_per_element)

    # KR: 일부 RGBA/LA 텍스처는 addrlib 기반 4KB_S 경로가 더 정확합니다.
    # EN: Some RGBA/LA textures are more accurate with addrlib-based 4KB_S path.
    if best_width == width and best_height == height and bytes_per_element in {2, 4}:
        addrlib_candidate = _ps5_unswizzle_addrlib_uncompressed_candidate(
            usable, width, height, bytes_per_element
        )
        if addrlib_candidate is not None:
            addrlib_data, addrlib_score = addrlib_candidate
            if addrlib_score < (best_score * 0.98):
                best_data = addrlib_data
                best_width = width
                best_height = height
                best_variant = "addrlib_4KB_S"
                best_score = addrlib_score

    # KR: Roughness guard – unswizzle 결과가 원본보다 거칠면, 원본이 이미 linear입니다.
    # EN: Roughness guard - if unswizzle result is rougher than original, original is already linear.
    if roughness_guard and raw_score is not None and best_score >= raw_score * 0.92:
        return clipped, width, height, "already_linear", raw_score

    return best_data, best_width, best_height, best_variant, best_score


def detect_ps5_swizzle_state_from_image(
    image: Image.Image,
    mask_x: int | None = None,
    mask_y: int | None = None,
    rotate: int = PS5_SWIZZLE_ROTATE,
) -> tuple[str, float, float, float]:
    """KR: Pillow 이미지의 swizzle 상태를 판별합니다.
    EN: Determine the swizzle state of a Pillow image.
    """
    prepared = _ps5_prepare_image(image)

    data = prepared.tobytes()
    bytes_per_element = len(prepared.getbands())
    verdict, raw_score, unsw_score, swz_score, _, _ = detect_ps5_swizzle_state(
        data,
        prepared.width,
        prepared.height,
        bytes_per_element,
        mask_x=mask_x,
        mask_y=mask_y,
    )
    return verdict, raw_score, unsw_score, swz_score


def apply_ps5_swizzle_to_image(
    image: Image.Image,
    mask_x: int | None = None,
    mask_y: int | None = None,
    rotate: int = PS5_SWIZZLE_ROTATE,
) -> Image.Image:
    """KR: 선형 이미지에 PS5 swizzle 변환을 적용합니다.
    EN: Apply PS5 swizzle transform to a linear image.
    """
    prepared = _ps5_prepare_image(image)
    bytes_per_element = len(prepared.getbands())
    if not _ps5_dimensions_supported(
        prepared.width, prepared.height, bytes_per_element
    ):
        return prepared
    # KR: 전치 bpe (예: Alpha8)에만 역방향 회전을 적용합니다.
    # EN: Apply inverse rotation only for transposed bpe (e.g. Alpha8).
    should_transpose = _PS5_AXIS_TRANSPOSE.get(bytes_per_element, False)
    if should_transpose and rotate % 360 != 0:
        prepared = prepared.rotate((-rotate) % 360, expand=True)
    if not _ps5_dimensions_supported(
        prepared.width, prepared.height, bytes_per_element
    ):
        return _ps5_prepare_image(image)

    data = prepared.tobytes()
    swizzled = ps5_swizzle_bytes(
        data,
        prepared.width,
        prepared.height,
        bytes_per_element,
        mask_x=mask_x,
        mask_y=mask_y,
    )
    return Image.frombytes(prepared.mode, (prepared.width, prepared.height), swizzled)


def apply_ps5_unswizzle_to_image(
    image: Image.Image,
    mask_x: int | None = None,
    mask_y: int | None = None,
    rotate: int = PS5_SWIZZLE_ROTATE,
    allow_axis_swap: bool = False,
    roughness_guard: bool = False,
) -> Image.Image:
    """KR: swizzled 이미지에 PS5 unswizzle 변환을 적용합니다.
    EN: Apply PS5 unswizzle transform to a swizzled image.
    roughness_guard=True이면, 이미 linear인 입력은 변환하지 않습니다.
    """
    prepared = _ps5_prepare_image(image)
    bytes_per_element = len(prepared.getbands())
    if not _ps5_dimensions_supported(
        prepared.width, prepared.height, bytes_per_element
    ):
        return prepared
    data = prepared.tobytes()
    unswizzled, out_width, out_height, variant, _ = _ps5_unswizzle_best_variant(
        data,
        prepared.width,
        prepared.height,
        bytes_per_element,
        mask_x=mask_x,
        mask_y=mask_y,
        allow_axis_swap=allow_axis_swap,
        roughness_guard=roughness_guard,
    )
    if variant == "already_linear":
        return prepared
    output = Image.frombytes(prepared.mode, (out_width, out_height), unswizzled)
    # KR: rotate는 축-스왑(전치) 된 경우에만 적용합니다 (예: Alpha8).
    # EN: Rotation is applied only when axis-swap (transpose) occurred (e.g. Alpha8).
    if variant == "swapped_axes" and rotate % 360 != 0:
        output = output.rotate(rotate % 360, expand=True)
    return output


def detect_texture_object_ps5_swizzle(
    texture_obj: Any,
    mask_x: int | None = None,
    mask_y: int | None = None,
    rotate: int = PS5_SWIZZLE_ROTATE,
) -> str | None:
    """KR: Texture2D 오브젝트의 swizzle 상태를 판별합니다.
    EN: Determines the swizzle state of a Texture2D object.
    """
    verdict, _ = detect_texture_object_ps5_swizzle_detail(
        texture_obj,
        mask_x=mask_x,
        mask_y=mask_y,
        rotate=rotate,
    )
    return verdict


def detect_texture_object_ps5_swizzle_detail(
    texture_obj: Any,
    mask_x: int | None = None,
    mask_y: int | None = None,
    rotate: int = PS5_SWIZZLE_ROTATE,
) -> tuple[str | None, str | None]:
    """KR: Texture2D 오브젝트의 swizzle 상태를 판별합니다.
    반환값은 (판정값, 판정근거)입니다.
    EN: Determines the swizzle state of a Texture2D object.
    Returns (verdict, reason).
    """
    try:
        texture = texture_obj.parse_as_object()
        width = int(getattr(texture, "m_Width", 0) or 0)
        height = int(getattr(texture, "m_Height", 0) or 0)
        stream_data = getattr(texture, "m_StreamData", None)
        try:
            stream_size = int(getattr(stream_data, "size", 0) or 0)
        except Exception:
            stream_size = 0
        is_readable = bool(getattr(texture, "m_IsReadable", False))
        try:
            texture_format = int(getattr(texture, "m_TextureFormat", -1) or -1)
        except Exception:
            texture_format = -1

        image_data = getattr(texture, "image_data", None)
        if isinstance(image_data, (bytes, bytearray)):
            image_data_len = len(image_data)
        else:
            image_data_len = 0

        # KR: 포맷/메타데이터 기반 공용 규칙:
        #  - BC: stream+non-readable => swizzled, inline+readable => linear
        #  - Crunched: UnityPy decode 경로 기준 linear 취급
        #  - Uncompressed: stream/inline 메타 + bpe 일치 여부로 판정
        # EN: Common rules based on format/metadata:
        #  - BC: stream+non-readable => swizzled, inline+readable => linear
        #  - Crunched: treated as linear per UnityPy decode path
        #  - Uncompressed: determined by stream/inline meta + bpe match
        meta_hint: str | None = None
        meta_source: str | None = None
        if width > 0 and height > 0:
            if _texture_format_is_crunched(texture_format):
                return "likely_linear_input", "crunched-unitypy-decode"

            expected_alpha8_size = width * height
            if (
                texture_format == 1
                and stream_size > 0
                and not is_readable
                and stream_size == expected_alpha8_size
            ):
                meta_hint = "likely_swizzled_input"
                meta_source = "meta-alpha8-stream"
            elif (
                texture_format == 1
                and stream_size == 0
                and not is_readable
                and image_data_len == expected_alpha8_size
            ):
                meta_hint = "likely_swizzled_input"
                meta_source = "meta-alpha8-inline-nonread"
            elif stream_size > 0 and not is_readable:
                meta_hint = "likely_swizzled_input"
                meta_source = "meta-stream"
            elif stream_size == 0 and is_readable and image_data_len > 0:
                meta_hint = "likely_linear_input"
                meta_source = "meta-inline"

        # KR: 메타 기준이 확실하면 유사도보다 우선합니다.
        # EN: If metadata criteria are definitive, they take priority over similarity.
        if meta_hint is not None:
            return meta_hint, meta_source or "meta"

        if width > 0 and height > 0:
            raw_data: bytes | None = None
            get_image_data = getattr(texture, "get_image_data", None)
            if callable(get_image_data):
                try:
                    candidate = get_image_data()
                    if isinstance(candidate, (bytes, bytearray)):
                        raw_data = bytes(candidate)
                except Exception:
                    raw_data = None
            if raw_data is None:
                image_data = getattr(texture, "image_data", None)
                if isinstance(image_data, (bytes, bytearray)):
                    raw_data = bytes(image_data)

            if raw_data:
                if _texture_format_is_bc(texture_format):
                    # KR: BC 포맷은 descriptor 비트(타일모드/selector)가 핵심이며
                    # 현재 자산 API에서 직접 노출되지 않으므로, 휴리스틱 점수 판별을 피합니다.
                    # EN: BC format relies on descriptor bits (tile mode/selector) which
                    # are not directly exposed by the current asset API, so heuristic scoring is avoided.
                    if stream_size > 0 and not is_readable:
                        return "likely_swizzled_input", "bc-meta-stream"
                    if stream_size == 0 and is_readable and image_data_len > 0:
                        return "likely_linear_input", "bc-meta-inline"
                    return "inconclusive", "bc-descriptor-unavailable"

                total_elements = width * height
                bytes_per_element: int | None = _texture_format_bytes_per_element(
                    texture_format
                )
                if (
                    bytes_per_element is None
                    and total_elements > 0
                    and (len(raw_data) % total_elements) == 0
                ):
                    derived = len(raw_data) // total_elements
                    if derived in {1, 2, 3, 4}:
                        bytes_per_element = int(derived)
                if bytes_per_element in {1, 2, 3, 4}:
                    expected_base = width * height * int(bytes_per_element)
                    if (
                        stream_size > 0
                        and not is_readable
                        and len(raw_data) >= expected_base
                    ):
                        return "likely_swizzled_input", "raw-meta-stream-bpe"
                    if stream_size == 0 and len(raw_data) >= expected_base:
                        return "likely_linear_input", "raw-meta-inline-bpe"

        image = getattr(texture, "image", None)
        if isinstance(image, Image.Image):
            verdict, _, _, _ = detect_ps5_swizzle_state_from_image(
                image,
                mask_x=mask_x,
                mask_y=mask_y,
                rotate=rotate,
            )
            return verdict, "image"
        return None, None
    except Exception:
        return None, None
