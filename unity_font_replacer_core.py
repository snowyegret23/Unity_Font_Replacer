"""KR: Unity 폰트 교체를 위한 핵심 CLI 및 처리 파이프라인.
이 모듈은 Unity 폰트 에셋의 스캔, 파싱, 교체, 프리뷰 내보내기,
    PS5 swizzle/unswizzle 지원 기능을 포함합니다.
주요 기능:
      - TTF 바이너리 교체: 기존 폰트 파일의 바이너리 데이터를 새 폰트로 대체
      - TMP SDF 폰트 데이터 변환: 구 스키마(old)와 신 스키마(new) 간 양방향 변환
      - 아틀라스 텍스처 교체: SDF 아틀라스 이미지 데이터 교체
      - 머티리얼 패칭: TMP 머티리얼의 셰이더 프로퍼티 패딩/스타일 보정
      - PS5 swizzle/unswizzle: PlayStation 5 텍스처 메모리 레이아웃 처리
      - 프리뷰 내보내기: 교체 전후 미리보기 이미지 출력
TMP 스키마 경계:
      - 구 스키마 (old): Unity <=2018.3.14, m_glyphInfoList 사용, top-origin Y 좌표계
      - 신/하이브리드 스키마: Unity >=2018.4.2, m_GlyphTable 사용, bottom-origin Y 좌표계
        (2018.4.2~2019.1은 legacy 필드도 함께 존재할 수 있음)

EN: Core CLI and processing pipeline for Unity font replacement.
This module includes scanning, parsing, replacement, preview export,
    and PS5 swizzle/unswizzle support for Unity font assets.
Key features:
      - TTF binary replacement: replace binary data of existing font files with new fonts
      - TMP SDF font data conversion: bidirectional conversion between old and new schemas
      - Atlas texture replacement: replace SDF atlas image data
      - Material patching: padding/style correction of TMP material shader properties
      - PS5 swizzle/unswizzle: PlayStation 5 texture memory layout handling
      - Preview export: output before/after preview images
TMP schema boundaries:
      - Old schema: Unity <=2018.3.14, uses m_glyphInfoList, top-origin Y coordinates
      - New/hybrid schema: Unity >=2018.4.2, uses m_GlyphTable, bottom-origin Y coordinates
        (2018.4.2~2019.1 may also retain legacy fields)
"""

from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import inspect
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback as tb_module
import copy
import errno
import struct
import weakref
from contextvars import ContextVar
from functools import lru_cache, wraps
from typing import Any, Callable, Iterable, Literal, NoReturn, cast

from PIL import Image, ImageOps
from asset_validation import (
    _collect_validation_inner_names,
    _structural_validate_unityfs_bundle,
)
from asset_scanner import (
    find_assets_files as _find_assets_files_impl,
    scan_fonts_from_env as _scan_fonts_from_env_impl,
    scan_fonts_in_asset_file as _scan_fonts_in_asset_file_impl,
)
from ps5_texture import (
    PS5_SWIZZLE_MASK_X as PS5_SWIZZLE_MASK_X,
    PS5_SWIZZLE_MASK_Y as PS5_SWIZZLE_MASK_Y,
    PS5_SWIZZLE_ROTATE,
    _PS5_BC_FORMATS,
    _ps5_decode_bc_to_rgba,
    _ps5_should_swap_rb_for_bc_preview,
    _ps5_swap_rb_image,
    _ps5_unswizzle_bc_best_layout_match,
    _ps5_unswizzle_best_variant,
    _texture_format_bytes_per_element,
    _texture_format_is_bc,
    apply_ps5_swizzle_to_image,
    apply_ps5_unswizzle_to_image,
    compute_ps5_swizzle_masks as compute_ps5_swizzle_masks,
    detect_ps5_swizzle_state as detect_ps5_swizzle_state,
    detect_ps5_swizzle_state_from_image as detect_ps5_swizzle_state_from_image,
    detect_texture_object_ps5_swizzle as detect_texture_object_ps5_swizzle,
    detect_texture_object_ps5_swizzle_detail,
    ps5_swizzle_bytes as ps5_swizzle_bytes,
    ps5_unswizzle_bytes as ps5_unswizzle_bytes,
)
from scan_worker_pool import (
    DEFAULT_MAX_JOBS_PER_WORKER,
    DEFAULT_STALL_SECONDS,
    PersistentScanWorkerPool,
    ScanPoolResult,
    decode_protocol_message,
    write_protocol_message,
)
from tmp_font_schema import (
    _TMP_CREATION_SETTINGS_KEYS as _TMP_CREATION_SETTINGS_KEYS,
    _atlas_ref_ids,
    _best_atlas_ref,
    _get_tmp_material_reference,
    _resolve_creation_settings_key,
    _sync_creation_settings_payload,
    _sync_existing_record_table,
    _sync_single_atlas_state,
    _tmp_flip_y_between_old_new,
    convert_face_info_new_to_old,
    convert_face_info_old_to_new as convert_face_info_old_to_new,
    convert_glyphs_new_to_old,
    convert_glyphs_old_to_new as convert_glyphs_old_to_new,
    detect_tmp_version as detect_tmp_version,
    ensure_int,
    extract_tmp_atlas_padding,
    inspect_tmp_font_schema,
    normalize_sdf_data,
)
from unitypy_runtime import (
    UnityPy,
    cleanup_unitypy_environments,
    close_unitypy_env,
    load_unitypy,
    missing_low_memory_features,
)
from UnityPy.files.replacers import Replacer
from UnityPy.helpers.TypeTreeGenerator import TypeTreeGenerator

try:
    from fontTools.ttLib import TTFont
except Exception:  # pragma: no cover - KR: 선택적 의존성 / EN: optional dependency
    TTFont = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


Language = Literal["ko", "en"]
JsonDict = dict[str, Any]
# KR: 등록된 임시 디렉토리 집합 (프로세스 종료 시 정리용)
# EN: Set of registered temp directories (cleaned up on process exit)
_REGISTERED_TEMP_DIRS: set[str] = set()
# KR: 텍스처 자동 분할 기준값: 단일 텍스처가 이 바이트 수를 초과하면 원샷 분할 적용
# EN: Auto-split threshold: apply one-shot split when a single texture exceeds this byte count
_AUTO_SPLIT_ONESHOT_TEXTURE_BYTES = 1536 * 1024 * 1024
# KR: 텍스처 배치 분할 목표 바이트 수
# EN: Texture batch split target byte count
_AUTO_SPLIT_TEXTURE_BATCH_TARGET_BYTES = 768 * 1024 * 1024


def _emit_phase_callback(
    phase_callback: Callable[[str, JsonDict], None] | None,
    phase: str,
    **payload: Any,
) -> None:
    """KR: 진행 단계 콜백을 안전하게 호출한다.
    매개변수:
        phase_callback: 호출할 콜백 함수 (None이면 무시)
        phase: 현재 처리 단계 이름
        **payload: 콜백에 전달할 추가 데이터

    EN: Safely invoke a progress phase callback.
    Args:
        phase_callback: 호출할 콜백 함수 (None이면 무시)
        phase: 현재 처리 단계 이름
        **payload: 콜백에 전달할 추가 데이터
    """
    if phase_callback is None:
        return
    try:
        phase_callback(phase, cast(JsonDict, payload))
    except Exception:
        logger.debug("단계 콜백 실패: %s", phase, exc_info=True)
# KR: TMP 더티 플래그 키 (룩업 테이블 재빌드 필요 여부)
# EN: TMP dirty flag key (whether lookup table rebuild is needed)
_TMP_DIRTY_FLAG_KEYS = (
    "m_IsFontAssetLookupTablesDirty",
    "IsFontAssetLookupTablesDirty",
)
# KR: TMP 글리프 인덱스 목록 키
# EN: TMP glyph index list keys
_TMP_GLYPH_INDEX_LIST_KEYS = (
    "m_GlyphIndexList",
    "m_GlyphIndexes",
)
# KR: Unity 에셋 번들 시그니처 문자열 집합
# EN: Unity asset bundle signature string set
BUNDLE_SIGNATURES = {"UnityFS", "UnityWeb", "UnityRaw"}
# KR: 구 스키마 라인 메트릭 키 목록 (TMP <=2018.3.14)
# EN: Old schema line metric key list (TMP <=2018.3.14)
_OLD_LINE_METRIC_KEYS = (
    "LineHeight",
    "Baseline",
    "Ascender",
    "CapHeight",
    "Descender",
    "CenterLine",
    "Scale",
    "SuperscriptOffset",
    "SubscriptOffset",
    "SubSize",
    "Underline",
    "UnderlineThickness",
    "strikethrough",
    "strikethroughThickness",
    "TabWidth",
)
# KR: 구 스키마에서 스케일 보정 대상이 되는 라인 메트릭 키
# EN: Old schema line metric keys subject to scale correction
_OLD_LINE_METRIC_SCALE_KEYS = (
    "LineHeight",
    "Baseline",
    "Ascender",
    "CapHeight",
    "Descender",
    "CenterLine",
    "SuperscriptOffset",
    "SubscriptOffset",
    "Underline",
    "UnderlineThickness",
    "strikethrough",
    "strikethroughThickness",
    "TabWidth",
)
# KR: 신 스키마 라인 메트릭 키 목록 (TMP >=2018.4.2)
# EN: New schema line metric key list (TMP >=2018.4.2)
_NEW_LINE_METRIC_KEYS = (
    "m_LineHeight",
    "m_AscentLine",
    "m_CapLine",
    "m_MeanLine",
    "m_Baseline",
    "m_DescentLine",
    "m_Scale",
    "m_SuperscriptOffset",
    "m_SuperscriptSize",
    "m_SubscriptOffset",
    "m_SubscriptSize",
    "m_UnderlineOffset",
    "m_UnderlineThickness",
    "m_StrikethroughOffset",
    "m_StrikethroughThickness",
    "m_TabWidth",
)
# KR: 신 스키마에서 스케일 보정 대상이 되는 라인 메트릭 키
# EN: New schema line metric keys subject to scale correction
_NEW_LINE_METRIC_SCALE_KEYS = (
    "m_LineHeight",
    "m_AscentLine",
    "m_CapLine",
    "m_MeanLine",
    "m_Baseline",
    "m_DescentLine",
    "m_SuperscriptOffset",
    "m_SubscriptOffset",
    "m_UnderlineOffset",
    "m_UnderlineThickness",
    "m_StrikethroughOffset",
    "m_StrikethroughThickness",
    "m_TabWidth",
)
# KR: 머티리얼 패딩 스케일 키: 아틀라스 크기 변경 시 비례 보정이 필요한 셰이더 프로퍼티
# EN: Material padding scale keys: shader properties needing proportional correction on atlas size change
_MATERIAL_PADDING_SCALE_KEYS = (
    "_GradientScale",
    "_FaceDilate",
    "_OutlineWidth",
    "_OutlineSoftness",
    "_UnderlayDilate",
    "_UnderlaySoftness",
    "_UnderlayOffsetX",
    "_UnderlayOffsetY",
    "_GlowOffset",
    "_GlowInner",
    "_GlowOuter",
)
# KR: 머티리얼 스타일 float 키: 원본 머티리얼의 시각적 스타일을 보존해야 하는 프로퍼티
# EN: Material style float keys: properties that must preserve the original material visual style
_MATERIAL_STYLE_FLOAT_KEYS = (
    "_FaceDilate",
    "_OutlineWidth",
    "_OutlineSoftness",
    "_UnderlayDilate",
    "_UnderlaySoftness",
    "_UnderlayOffsetX",
    "_UnderlayOffsetY",
    "_GlowOffset",
    "_GlowInner",
    "_GlowOuter",
    "_ScaleRatioA",
    "_ScaleRatioB",
    "_ScaleRatioC",
)
# KR: 머티리얼 스타일에서 패딩 스케일 보정이 필요한 키
# EN: Material style keys needing padding scale correction
_MATERIAL_STYLE_PADDING_SCALE_KEYS = (
    "_FaceDilate",
    "_OutlineWidth",
    "_OutlineSoftness",
    "_UnderlayDilate",
    "_UnderlaySoftness",
    "_UnderlayOffsetX",
    "_UnderlayOffsetY",
    "_GlowOffset",
    "_GlowInner",
    "_GlowOuter",
)
# KR: 머티리얼 스타일 색상 키: 원본에서 보존해야 하는 색상 프로퍼티
# EN: Material style color keys: color properties to preserve from original
_MATERIAL_STYLE_COLOR_KEYS = (
    "_FaceColor",
    "_OutlineColor",
    "_UnderlayColor",
    "_GlowColor",
)
# KR: 외곽선 비율 보정 대상 키
# EN: Outline ratio correction target keys
_MATERIAL_OUTLINE_RATIO_KEYS = (
    "_OutlineWidth",
    "_OutlineSoftness",
)
# KR: 로그 포맷 상수
# EN: Log format constants
LOG_CONSOLE_FORMAT = "%(message)s"  # KR: 콘솔 출력 포맷 (메시지만) / EN: Console output format (message only)
LOG_FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"  # KR: 파일 로그 포맷 / EN: File log format
LOG_FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"  # KR: 파일 로그 날짜 포맷 / EN: File log date format
VERBOSE_LOG_FILENAME = "verbose.txt"  # KR: 상세 로그 파일명 / EN: Verbose log filename


def _compose_log_message(*parts: object, sep: str = " ") -> str:
    """KR: 로그 파트들을 하나의 문자열로 합친다.
    매개변수:
        *parts: 로그 메시지를 구성하는 각 부분
        sep: 구분자 (기본: 공백)
    반환값:
        합쳐진 로그 메시지 문자열

    EN: Combine log parts into a single string.
    Args:
        *parts: 로그 메시지를 구성하는 각 부분
        sep: 구분자 (기본: 공백)
    Returns:
        합쳐진 로그 메시지 문자열
    """
    return sep.join(str(part) for part in parts)


def _configure_logging(
    console_level: int = logging.INFO,
    verbose_log_path: str | None = None,
) -> None:
    """KR: 콘솔 및 선택적 파일 로그 핸들러를 구성한다.
    매개변수:
        console_level: 콘솔 출력 로그 레벨 (기본: INFO)
        verbose_log_path: 상세 로그 파일 경로 (None이면 파일 로그 비활성화)

    EN: Configure console and optional file log handlers.
    Args:
        console_level: 콘솔 출력 로그 레벨 (기본: INFO)
        verbose_log_path: 상세 로그 파일 경로 (None이면 파일 로그 비활성화)
    """
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG if verbose_log_path else console_level)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(LOG_CONSOLE_FORMAT))
    root_logger.addHandler(console_handler)

    if verbose_log_path:
        file_handler = logging.FileHandler(
            verbose_log_path,
            mode="w",
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(LOG_FILE_FORMAT, datefmt=LOG_FILE_DATE_FORMAT)
        )
        root_logger.addHandler(file_handler)


def _coerce_log_level(message: str, default_level: int = logging.INFO) -> int:
    """KR: 지역화된 메시지 접두사로부터 로그 레벨을 추론한다.
    매개변수:
        message: 로그 메시지 문자열
        default_level: 추론 실패 시 기본 레벨 (기본: INFO)
    반환값:
        추론된 로그 레벨 정수

    EN: Infer log level from localized message prefix.
    Args:
        message: 로그 메시지 문자열
        default_level: 추론 실패 시 기본 레벨 (기본: INFO)
    Returns:
        추론된 로그 레벨 정수
    """
    lowered = message.lower()
    if "경고" in message or "warning" in lowered:
        return logging.WARNING
    if (
        "오류" in message
        or "error" in lowered
        or "failed" in lowered
        or "실패" in message
    ):
        return logging.ERROR
    return default_level


def _log_console(
    *parts: object,
    sep: str = " ",
    level: int | None = None,
    include_traceback: bool = False,
) -> None:
    """KR: 레거시 호출 지점에서 사용하는 print 호환 로깅 브리지.
    매개변수:
        *parts: 로그 메시지 부분들
        sep: 구분자
        level: 로그 레벨 (None이면 메시지 내용에서 자동 추론)
        include_traceback: True이면 예외 Traceback 포함

    EN: Print-compatible logging bridge used at legacy call sites.
    Args:
        *parts: 로그 메시지 부분들
        sep: 구분자
        level: 로그 레벨 (None이면 메시지 내용에서 자동 추론)
        include_traceback: True이면 예외 Traceback 포함
    """
    message = _compose_log_message(*parts, sep=sep)
    resolved_level = _coerce_log_level(message) if level is None else level
    if include_traceback:
        logger.log(resolved_level, message, exc_info=True)
        return
    logger.log(resolved_level, message)


def _log_debug(*parts: object, sep: str = " ") -> None:
    """KR: 디버그 레벨 로그를 기록한다.
    EN: Record a debug-level log entry.
    """
    logger.debug(_compose_log_message(*parts, sep=sep))


def _log_info(*parts: object, sep: str = " ") -> None:
    """KR: 정보 레벨 로그를 기록한다.
    EN: Record an info-level log entry.
    """
    logger.info(_compose_log_message(*parts, sep=sep))


def _log_warning(*parts: object, sep: str = " ") -> None:
    """KR: 경고 레벨 로그를 기록한다.
    EN: Record a warning-level log entry.
    """
    logger.warning(_compose_log_message(*parts, sep=sep))


def _log_exception(*parts: object, sep: str = " ") -> None:
    """KR: 예외 Traceback을 포함한 에러 로그를 기록한다.
    EN: Record an error log entry including exception traceback.
    """
    logger.exception(_compose_log_message(*parts, sep=sep))


def find_ggm_file(data_path: str) -> str | None:
    """KR: 데이터 폴더에서 globalgamemanagers 계열 파일 경로를 찾는다.
    EN: Find the globalgamemanagers family file path in the data folder.
    """
    candidates = ["globalgamemanagers", "globalgamemanagers.assets", "data.unity3d"]
    candidates_resources = ["unity default resources", "unity_builtin_extra"]
    fls: list[str] = []
    # KR: globalgamemanagers 핵심 파일을 우선 탐색한다
    # EN: Search for globalgamemanagers core files first
    for candidate in candidates:
        ggm_path = os.path.join(data_path, candidate)
        if os.path.exists(ggm_path):
            fls.append(ggm_path)
    for candidate in candidates_resources:
        ggm_path = os.path.join(data_path, "Resources", candidate)
        if os.path.exists(ggm_path):
            fls.append(ggm_path)
    if fls:
        return fls[0]
    return None


def resolve_game_path(path: str, lang: Language = "ko") -> tuple[str, str]:
    """KR: 입력 경로를 게임 루트와 _Data 경로로 정규화한다.
    EN: Normalize the input path to game root and _Data path.
    """
    path = os.path.normpath(os.path.abspath(path))

    if path.lower().endswith("_data"):
        data_path = path
        game_path = os.path.dirname(path)
    else:
        game_path = path
        data_folders = [
            d
            for d in os.listdir(path)
            if d.lower().endswith("_data") and os.path.isdir(os.path.join(path, d))
        ]

        if not data_folders:
            if lang == "ko":
                raise FileNotFoundError(f"'{path}'에서 _Data 폴더를 찾을 수 없습니다.")
            raise FileNotFoundError(f"Could not find _Data folder in '{path}'.")

        data_path = os.path.join(game_path, data_folders[0])

    ggm_path = find_ggm_file(data_path)
    if not ggm_path:
        if lang == "ko":
            raise FileNotFoundError(
                f"'{data_path}'에서 globalgamemanagers 파일을 찾을 수 없습니다.\n올바른 Unity 게임 폴더인지 확인해주세요."
            )
        raise FileNotFoundError(
            f"Could not find a globalgamemanagers file in '{data_path}'.\nPlease verify this is a valid Unity game folder."
        )

    return game_path, data_path


def get_data_path(game_path: str, lang: Language = "ko") -> str:
    """KR: 게임 루트에서 _Data 폴더 경로를 반환한다.
    EN: Return the _Data folder path from the game root.
    """
    data_folders = [i for i in os.listdir(game_path) if i.lower().endswith("_data")]
    if not data_folders:
        if lang == "ko":
            raise FileNotFoundError(f"'{game_path}'에서 _Data 폴더를 찾을 수 없습니다.")
        raise FileNotFoundError(f"Could not find _Data folder in '{game_path}'.")
    return os.path.join(game_path, data_folders[0])


def get_unity_version(game_path: str, lang: Language = "ko") -> str:
    """KR: 게임 경로에서 Unity 버전을 읽어 반환한다.
    EN: Read and return the Unity version from the game path.
    """
    data_path = get_data_path(game_path, lang=lang)
    candidates = [
        os.path.join(data_path, "globalgamemanagers"),
        os.path.join(data_path, "globalgamemanagers.assets"),
        os.path.join(data_path, "data.unity3d"),
    ]
    existing_candidates = [p for p in candidates if os.path.exists(p)]
    if not existing_candidates:
        if lang == "ko":
            raise FileNotFoundError(
                f"'{data_path}'에서 globalgamemanagers 파일을 찾을 수 없습니다.\n올바른 Unity 게임 폴더인지 확인해주세요."
            )
        raise FileNotFoundError(
            f"Could not find a globalgamemanagers file in '{data_path}'.\nPlease verify this is a valid Unity game folder."
        )

    for candidate in existing_candidates:
        env = None
        try:
            env = load_unitypy(candidate)

            # KR: 1) 빠른 경로: 최상위 파일에서 unity_version을 바로 확인한다
            # EN: 1) Fast path: check unity_version directly on top-level file
            top_file = getattr(env, "file", None)
            top_version = getattr(top_file, "unity_version", None)
            if top_version:
                return str(top_version)

            # KR: 2) 로드된 파일들을 확인한다
            # EN: 2) Check loaded files
            env_files = getattr(env, "files", None)
            if isinstance(env_files, dict):
                for loaded in env_files.values():
                    uv = getattr(loaded, "unity_version", None)
                    if uv:
                        return str(uv)

            # KR: 3) 폴백: 파싱된 오브젝트가 있을 때만 검사한다
            # EN: 3) Fallback: inspect only when parsed objects exist
            objs = getattr(env, "objects", None)
            if objs:
                first_obj = objs[0]
                assets_file = getattr(first_obj, "assets_file", None)
                uv = getattr(assets_file, "unity_version", None)
                if uv:
                    return str(uv)
        except Exception:
            continue
        finally:
            close_unitypy_env(env)
            env = None
            gc.collect()

    tried = ", ".join(os.path.basename(p) for p in existing_candidates)
    if lang == "ko":
        raise RuntimeError(f"Unity 버전 감지에 실패했습니다. 시도한 파일: {tried}")
    raise RuntimeError(f"Failed to detect Unity version. Tried files: {tried}")


def get_script_dir() -> str:
    """KR: 실행 기준 디렉터리(스크립트/배포 바이너리)를 반환한다.
    EN: Return the execution base directory (script/distribution binary).
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def parse_target_files_arg(target_file_args: list[str] | None) -> set[str]:
    """KR: --target-file 인자(반복/콤마 구분)를 파일명 집합으로 정규화한다.
    EN: Normalize --target-file arguments (repeated/comma-separated) into a filename set.
    """
    selected_files: set[str] = set()
    if not target_file_args:
        return selected_files
    for entry in target_file_args:
        for token in str(entry).split(","):
            name = os.path.basename(token.strip())
            if name:
                selected_files.add(name)
    return selected_files


def parse_exclude_exts_arg(exclude_ext_args: list[str] | None) -> set[str]:
    """KR: --exclude-ext 인자(반복/콤마 구분)를 확장자 집합으로 정규화한다.
    EN: Normalize --exclude-ext arguments (repeated/comma-separated) into an extension set.
    """
    normalized_exts: set[str] = set()
    if not exclude_ext_args:
        return normalized_exts
    for entry in exclude_ext_args:
        for token in str(entry).split(","):
            raw = token.strip().lower()
            if not raw:
                continue
            if raw.startswith("*"):
                raw = raw.lstrip("*")
            if not raw:
                continue
            if not raw.startswith("."):
                raw = f".{raw}"
            normalized_exts.add(raw)
    return normalized_exts


_PRIMARY_MODE_ARGS: tuple[tuple[str, str], ...] = (
    ("parse", "--parse"),
    ("mulmaru", "--mulmaru"),
    ("nanumgothic", "--nanumgothic"),
    ("font", "--font"),
    ("list", "--list"),
    ("preview_export", "--preview-export"),
)


def _selected_primary_modes(args: Any) -> list[str]:
    """KR: CLI 인자에서 활성화된 주요 모드 목록을 반환한다.
    EN: Return the list of active primary modes from CLI arguments.
    """
    selected: list[str] = []
    for attr_name, cli_name in _PRIMARY_MODE_ARGS:
        value = getattr(args, attr_name, None)
        if isinstance(value, str):
            if value.strip():
                selected.append(cli_name)
        elif value:
            selected.append(cli_name)
    return selected


def _mode_uses_scan_jobs(mode: str | None) -> bool:
    """KR: 해당 모드가 스캔 작업을 사용하는지 여부를 반환한다.
    EN: Return whether the given mode uses scan jobs.
    """
    return mode in {"parse", "mulmaru", "nanumgothic", "font", "preview_export"}


def _should_pause_before_exit(*, interactive_session: bool = False) -> bool:
    """KR: 종료 전 일시정지가 필요한지 판별한다.
    EN: Determine whether a pause is needed before exit.
    """
    return bool(interactive_session or getattr(sys, "frozen", False))


def _pause_before_exit(
    lang: Language = "ko",
    *,
    interactive_session: bool = False,
) -> None:
    """KR: 대화형 세션 또는 배포 바이너리 실행 시 종료 전 사용자 입력을 대기한다.
    EN: Wait for user input before exit in interactive session or distribution binary execution.
    """
    if not _should_pause_before_exit(interactive_session=interactive_session):
        return
    if lang == "ko":
        input("\n엔터를 눌러 종료...")
    else:
        input("\nPress Enter to exit...")


def strip_wrapping_quotes_repeated(value: str) -> str:
    """KR: 앞뒤 따옴표(' 또는 ")를 반복 제거한다.
    EN: Repeatedly strip leading/trailing quotes (' or ").
    """
    text = str(value).strip()
    while True:
        updated = text.strip().strip('"').strip("'")
        if updated == text:
            return updated
        text = updated


def sanitize_filename_component(
    value: str, fallback: str = "unnamed", max_len: int = 96
) -> str:
    """KR: 파일명 구성요소에서 경로/예약 문자를 안전한 문자로 치환한다.
    EN: Replace path/reserved characters with safe characters in a filename component.
    """
    text = str(value or "").strip()
    invalid_chars = '<>:"/\\|?*'
    cleaned = "".join("_" if ch in invalid_chars else ch for ch in text)
    cleaned = cleaned.strip().strip(".")
    if not cleaned:
        cleaned = fallback
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]
    return cleaned


def resolve_output_only_path(source_file: str, data_path: str, output_root: str) -> str:
    """KR: output-only 저장 시 원본 data_path 기준 상대 경로를 유지한 출력 경로를 계산한다.
    EN: Compute the output path preserving relative path from original data_path for output-only saves.
    """
    source_abs = os.path.abspath(source_file)
    data_abs = os.path.abspath(data_path)
    output_abs = os.path.abspath(output_root)
    try:
        rel_path = os.path.relpath(source_abs, data_abs)
    except ValueError:
        rel_path = os.path.basename(source_abs)
    if rel_path.startswith("..") or os.path.isabs(rel_path):
        rel_path = os.path.basename(source_abs)
    return os.path.join(output_abs, rel_path)


def prepare_output_only_dependencies(
    data_path: str,
    output_root: str,
    lang: Language = "ko",
    transaction: _DeferredPatchTransaction | None = None,
) -> None:
    """KR: output-only 모드에서 핵심 의존 파일을 출력 루트에 미리 복사한다.
    EN: Pre-copy essential dependency files to the output root in output-only mode.
    """
    candidate_rel_paths = [
        "globalgamemanagers",
        "globalgamemanagers.assets",
        "data.unity3d",
        os.path.join("Resources", "unity default resources"),
        os.path.join("Resources", "unity_builtin_extra"),
    ]
    copied: list[str] = []
    for rel_path in candidate_rel_paths:
        source_path = os.path.join(data_path, rel_path)
        if not os.path.isfile(source_path):
            continue
        output_path = os.path.join(output_root, rel_path)
        if os.path.exists(output_path):
            continue
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        if transaction is not None:
            transaction.backup(output_path, allow_missing=True)
        shutil.copy2(source_path, output_path)
        copied.append(rel_path)

    if copied:
        if lang == "ko":
            _log_console(
                f"출력 전용 의존 파일 준비: {len(copied)}개 ({', '.join(copied)})"
            )
        else:
            _log_console(
                f"Prepared output-only dependencies: {len(copied)} ({', '.join(copied)})"
            )


def register_temp_dir_for_cleanup(path: str) -> str:
    """KR: 종료 시 삭제할 임시 디렉터리를 등록하고 정규화 경로를 반환한다.
    EN: Register a temp directory for cleanup on exit and return the normalized path.
    """
    normalized = os.path.abspath(path)
    _REGISTERED_TEMP_DIRS.add(normalized)
    return normalized


def cleanup_registered_temp_dirs() -> None:
    """KR: 등록된 임시 디렉터리를 깊은 경로부터 안전하게 삭제한다.
    EN: Safely delete registered temp directories starting from deepest paths.
    """
    if not _REGISTERED_TEMP_DIRS:
        return
    for temp_dir in sorted(_REGISTERED_TEMP_DIRS, key=len, reverse=True):
        try:
            if os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception:
            pass
    _REGISTERED_TEMP_DIRS.clear()


atexit.register(cleanup_registered_temp_dirs)


def _atomic_replace_validated_file(source: str, destination: str) -> None:
    """Atomically install a validated file, including cross-volume temp roots."""
    try:
        os.replace(source, destination)
        return
    except OSError as error:
        if error.errno != errno.EXDEV and getattr(error, "winerror", None) != 17:
            raise

    destination_dir = os.path.dirname(os.path.abspath(destination)) or os.curdir
    os.makedirs(destination_dir, exist_ok=True)
    fd, staged_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.",
        suffix=".validated.tmp",
        dir=destination_dir,
    )
    try:
        with open(source, "rb") as source_stream, os.fdopen(fd, "wb") as staged:
            fd = -1
            shutil.copyfileobj(source_stream, staged, length=1024 * 1024)
            staged.flush()
            os.fsync(staged.fileno())
        try:
            shutil.copystat(source, staged_path)
        except OSError:
            pass
        os.replace(staged_path, destination)
        try:
            os.remove(source)
        except OSError:
            pass
    except BaseException:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.remove(staged_path)
        except OSError:
            pass
        raise


def _hash_file_contents(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hash_pil_image_rows(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(f"{image.mode}:{image.width}x{image.height}".encode("ascii"))
    for row in range(image.height):
        with image.crop((0, row, image.width, row + 1)) as row_image:
            digest.update(row_image.tobytes())
    return digest.hexdigest()


def _deferred_patch_fingerprint(patch_kind: str, payload: Any) -> str:
    """Build a stable content fingerprint without loading spilled atlases into RAM."""
    if not isinstance(payload, dict):
        return hashlib.sha256(repr(payload).encode("utf-8", errors="replace")).hexdigest()
    ignored = {
        "source_entry",
        "font_name",
        "replacement_font",
        "preview_sdf_data",
        "source_atlas",
        "source_atlas_path",
        "alpha8_linear_source",
        "alpha8_linear_source_path",
    }
    stable_payload = {key: value for key, value in payload.items() if key not in ignored}
    digest = hashlib.sha256()
    digest.update(str(patch_kind).encode("utf-8"))
    digest.update(
        json.dumps(
            stable_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=repr,
        ).encode("utf-8")
    )
    for image_key, path_key in (
        ("source_atlas", "source_atlas_path"),
        ("alpha8_linear_source", "alpha8_linear_source_path"),
    ):
        path = str(payload.get(path_key, "") or "").strip()
        image = payload.get(image_key)
        if path and os.path.isfile(path):
            image_hash = _hash_file_contents(path)
        elif isinstance(image, Image.Image):
            image_hash = _hash_pil_image_rows(image)
        else:
            image_hash = "missing"
        digest.update(f"{image_key}:{image_hash}".encode("ascii"))
    return digest.hexdigest()


class DeferredPatchAtomicityError(RuntimeError):
    """Raised when a cross-file TMP patch cannot be completed atomically."""


class _DeferredPatchTransaction:
    """Disk-backed rollback for a group of cross-file TMP patches."""

    def __init__(self, backup_root: str | None = None) -> None:
        normalized_root = (
            os.path.abspath(backup_root) if backup_root is not None else None
        )
        self._remove_backup_root_when_empty = bool(
            normalized_root is not None and not os.path.exists(normalized_root)
        )
        self._backup_root = normalized_root
        if normalized_root is not None:
            os.makedirs(normalized_root, exist_ok=True)
        self._backup_dir = tempfile.mkdtemp(
            prefix="unity_font_replacer_rollback_",
            dir=normalized_root,
        )
        self._backups: dict[str, str] = {}
        self._plan_fingerprints: dict[tuple[str, str, str], str] = {}
        self._payload_fingerprints: dict[tuple[int, str], tuple[Any, str]] = {}
        self._plan_conflicts: list[str] = []
        self._failures: list[str] = []
        self._active = True
        atexit.register(self.rollback)

    @property
    def has_backups(self) -> bool:
        return bool(self._backups)

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def backup_count(self) -> int:
        return len(self._backups)

    @property
    def backup_directory(self) -> str:
        return self._backup_dir

    @property
    def has_conflicts(self) -> bool:
        return bool(self._plan_conflicts)

    @property
    def has_failures(self) -> bool:
        return bool(self._failures or self._plan_conflicts)

    def fail(self, reason: str) -> None:
        if reason not in self._failures:
            self._failures.append(reason)

    def register_plan(
        self,
        patch_kind: str,
        target_file_key: str,
        object_key: str,
        payload: Any,
    ) -> bool:
        normalized_file = _normalize_asset_file_key(target_file_key) or str(
            target_file_key
        )
        plan_key = (str(patch_kind), normalized_file, str(object_key).lower())
        payload_cache_key = (id(payload), str(patch_kind))
        cached_payload = self._payload_fingerprints.get(payload_cache_key)
        if cached_payload is not None and cached_payload[0] is payload:
            fingerprint = cached_payload[1]
        else:
            fingerprint = _deferred_patch_fingerprint(patch_kind, payload)
            self._payload_fingerprints[payload_cache_key] = (payload, fingerprint)
        previous = self._plan_fingerprints.get(plan_key)
        if previous is None:
            self._plan_fingerprints[plan_key] = fingerprint
            return True
        if previous == fingerprint:
            return True
        conflict = (
            f"kind={patch_kind} file={normalized_file} key={object_key} "
            f"previous={previous} new={fingerprint}"
        )
        self._plan_conflicts.append(conflict)
        _log_warning(f"[deferred_transaction] conflicting target plan: {conflict}")
        return False

    def backup(
        self,
        destination: str,
        *,
        allow_missing: bool = False,
        replace_only: bool = False,
    ) -> None:
        if not self._active:
            raise RuntimeError("deferred patch transaction is no longer active")
        normalized = os.path.normcase(os.path.abspath(destination))
        if normalized in self._backups:
            return
        if not os.path.exists(destination):
            if allow_missing:
                self._backups[normalized] = ""
                return
            raise FileNotFoundError(destination)
        if not os.path.isfile(destination):
            raise IsADirectoryError(destination)

        fd, backup_path = tempfile.mkstemp(
            prefix=f"{len(self._backups):04d}_",
            suffix=".rollback",
            dir=self._backup_dir,
        )
        try:
            os.close(fd)
            fd = -1
            os.remove(backup_path)
            try:
                if not replace_only:
                    raise OSError(errno.ENOTSUP, "copy-safe snapshot required")
                # Replacing the destination later leaves this original inode
                # intact, so same-volume snapshots consume almost no space.
                os.link(destination, backup_path)
            except OSError:
                with open(destination, "rb") as source_stream, open(
                    backup_path, "xb"
                ) as backup_stream:
                    shutil.copyfileobj(
                        source_stream, backup_stream, length=1024 * 1024
                    )
                    backup_stream.flush()
                    os.fsync(backup_stream.fileno())
                try:
                    shutil.copystat(destination, backup_path)
                except OSError:
                    pass
            self._backups[normalized] = backup_path
        except BaseException:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                os.remove(backup_path)
            except OSError:
                pass
            raise

    def _remove_backup_directory(self) -> bool:
        try:
            shutil.rmtree(self._backup_dir)
        except FileNotFoundError:
            return True
        except Exception as error:
            _log_warning(
                f"[deferred_transaction] could not remove backup directory "
                f"{self._backup_dir}: {type(error).__name__}: {error}"
            )
            return False
        if self._remove_backup_root_when_empty and self._backup_root:
            try:
                os.rmdir(self._backup_root)
            except OSError:
                pass
        return True

    def commit(self) -> None:
        if not self._active:
            return
        self._active = False
        self._plan_fingerprints.clear()
        self._payload_fingerprints.clear()
        self._plan_conflicts.clear()
        self._failures.clear()
        if self._remove_backup_directory():
            self._backups.clear()

    def rollback(self) -> bool:
        if not self._active:
            return True
        restored = True
        for destination, backup_path in reversed(list(self._backups.items())):
            if not backup_path:
                try:
                    if os.path.isfile(destination):
                        os.remove(destination)
                    self._backups.pop(destination, None)
                except Exception as error:
                    restored = False
                    _log_warning(
                        f"[deferred_transaction] rollback delete failed: {destination}: "
                        f"{type(error).__name__}: {error}"
                    )
                continue
            if not os.path.isfile(backup_path):
                restored = False
                _log_warning(
                    f"[deferred_transaction] rollback backup missing: {backup_path}"
                )
                continue
            try:
                _atomic_replace_validated_file(backup_path, destination)
                self._backups.pop(destination, None)
            except Exception as error:
                restored = False
                _log_warning(
                    f"[deferred_transaction] rollback failed: {destination}: "
                    f"{type(error).__name__}: {error}"
                )
        if restored:
            self._active = False
            self._backups.clear()
            self._plan_fingerprints.clear()
            self._payload_fingerprints.clear()
            self._plan_conflicts.clear()
            self._failures.clear()
            self._remove_backup_directory()
        return restored


_ACTIVE_DEFERRED_TRANSACTION: ContextVar[_DeferredPatchTransaction | None] = (
    ContextVar("unity_font_replacer_deferred_transaction", default=None)
)


def _rollback_deferred_transaction_on_exit(
    func: Callable[..., Any],
) -> Callable[..., Any]:
    """Immediately roll back an unfinished CLI transaction on every exit path."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        token = _ACTIVE_DEFERRED_TRANSACTION.set(None)
        try:
            return func(*args, **kwargs)
        finally:
            transaction = _ACTIVE_DEFERRED_TRANSACTION.get()
            if transaction is not None and transaction.is_active:
                if not transaction.rollback():
                    _log_warning(
                        "[deferred_transaction] automatic rollback failed; "
                        f"backups remain at {transaction.backup_directory}"
                    )
            _ACTIVE_DEFERRED_TRANSACTION.reset(token)

    return wrapper


def normalize_font_name(name: str) -> str:
    """KR: 확장자/SDF 접미사를 제거해 폰트 기본 이름으로 정규화한다.
    EN: Normalize to the base font name by removing extensions/SDF suffixes.
    """
    for ext in [".ttf", ".otf", ".json", ".png"]:
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
    for suffix in (
        " SDF Atlas",
        " Raster Atlas",
        " Atlas",
        " SDF Material",
        " Raster Material",
        " Material",
        " SDF",
        " Raster",
    ):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def parse_bool_flag(value: Any) -> bool:
    """KR: 문자열/숫자/불리언 입력을 안전하게 bool로 해석한다.
    EN: Safely interpret string/number/boolean input as bool.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _read_bundle_signature(
    path: str, bundle_signatures: set[str] | None = None
) -> str | None:
    """KR: 파일 헤더에서 Unity 번들 시그니처를 읽는다.
    EN: Read a Unity bundle signature from the file header.
    """
    signatures = bundle_signatures or BUNDLE_SIGNATURES
    try:
        with open(path, "rb") as f:
            header = f.read(16)
    except Exception:
        return None

    for sig in signatures:
        token = (sig + "\x00").encode("ascii")
        if header.startswith(token):
            return sig
    return None


def _safe_metric_scale(game_point_size: Any, replacement_point_size: Any) -> float:
    """KR: 게임 pointSize 대비 교체 pointSize 비율을 계산한다.
    EN: Compute the ratio of replacement pointSize to game pointSize.
    """
    try:
        game_ps = float(game_point_size)
        repl_ps = float(replacement_point_size)
        if game_ps > 0 and repl_ps > 0:
            return repl_ps / game_ps
    except Exception:
        pass
    return 1.0


def _detect_target_texture_swizzle(
    texture_object_lookup: dict[tuple[str, int], Any],
    texture_swizzle_state_cache: dict[str, tuple[str | None, str | None]],
    assets_name: str,
    path_id: int,
) -> tuple[str | None, str | None]:
    """KR: 타겟 Texture2D의 swizzle 판정 결과를 캐시와 함께 반환합니다.
    EN: Return the swizzle detection result for the target Texture2D, with caching.
    """
    cache_key = f"{assets_name}|{path_id}"
    if cache_key in texture_swizzle_state_cache:
        return texture_swizzle_state_cache[cache_key]
    texture_obj = texture_object_lookup.get((assets_name, int(path_id)))
    verdict, source = (
        detect_texture_object_ps5_swizzle_detail(texture_obj)
        if texture_obj is not None
        else (None, None)
    )
    texture_swizzle_state_cache[cache_key] = (verdict, source)
    return verdict, source


def _preview_visible_image(image: Image.Image) -> Image.Image:
    """KR: RGBA/LA Atlas를 사람이 보기 쉬운 단일 채널 이미지로 정규화합니다.
    EN: Normalize an RGBA/LA atlas to a human-viewable single-channel image.
    """
    try:
        if image.mode == "RGBA":
            alpha = image.getchannel("A")
            rgb = image.convert("RGB")
            rgb_bbox = rgb.getbbox()
            alpha_bbox = alpha.getbbox()
            if alpha_bbox and not rgb_bbox:
                return alpha
            return alpha if alpha_bbox else image.convert("L")
        if image.mode == "LA":
            alpha = image.getchannel("A")
            return alpha if alpha.getbbox() else image.getchannel("L")
        if image.mode == "P":
            return image.convert("L")
        if image.mode not in {"L", "RGB"}:
            return image.convert("L")
        return image
    except Exception:
        return image.convert("L")


def _load_target_unswizzled_preview_image(
    texture_object_lookup: dict[tuple[str, int], Any],
    assets_name: str,
    atlas_path_id: int,
    swizzle_verdict: str | None,
    preview_rotate: int = PS5_SWIZZLE_ROTATE,
) -> Image.Image | None:
    """KR: 대상 게임 Atlas(Texture2D)에서 검증용 unswizzle preview 이미지를 생성합니다.
    EN: Generate an unswizzled preview image from the target game atlas (Texture2D) for verification.
    """
    texture_obj = texture_object_lookup.get((assets_name, int(atlas_path_id)))
    if texture_obj is None:
        return None
    try:
        texture = texture_obj.parse_as_object()
        width = int(getattr(texture, "m_Width", 0) or 0)
        height = int(getattr(texture, "m_Height", 0) or 0)
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

        if width > 0 and height > 0 and raw_data:
            total_elements = width * height
            bpe: int | None = None
            try:
                texture_format = int(getattr(texture, "m_TextureFormat", -1) or -1)
            except Exception:
                texture_format = -1

            if _texture_format_is_bc(texture_format):
                bc_info = _PS5_BC_FORMATS.get(texture_format)
                if bc_info is not None:
                    block_w_px, block_h_px, bytes_per_block, _ = bc_info
                    logical_block_w = (width + block_w_px - 1) // block_w_px
                    logical_block_h = (height + block_h_px - 1) // block_h_px
                    logical_bytes = (
                        logical_block_w * logical_block_h * bytes_per_block
                    )
                    candidate_raw = raw_data[:logical_bytes]
                    best = None
                    if swizzle_verdict != "likely_linear_input":
                        mip_count = int(getattr(texture, "m_MipCount", 1) or 1)
                        best = _ps5_unswizzle_bc_best_layout_match(
                            raw_data,
                            width,
                            height,
                            texture_format,
                            mip_count=mip_count,
                        )
                    if best is not None:
                        best_raw, _, _, _, _ = best
                        if swizzle_verdict == "likely_swizzled_input":
                            candidate_raw = best_raw
                    rgba = _ps5_decode_bc_to_rgba(
                        candidate_raw, width, height, texture_format
                    )
                    if rgba is not None:
                        preview_rgba = Image.frombytes("RGBA", (width, height), rgba)
                        if _ps5_should_swap_rb_for_bc_preview(texture_format):
                            preview_rgba = _ps5_swap_rb_image(preview_rgba)
                        # KR: BC preview는 Unity 좌표계와 일치하도록 상하 반전
                        # EN: Flip vertically to match Unity coordinate system for BC preview
                        return ImageOps.flip(preview_rgba)

            bpe_hint = _texture_format_bytes_per_element(texture_format)
            if bpe_hint is not None:
                bpe = bpe_hint
            elif total_elements > 0 and (len(raw_data) % total_elements) == 0:
                derived_bpe = len(raw_data) // total_elements
                if derived_bpe in {1, 2, 3, 4}:
                    bpe = derived_bpe

            if bpe in {1, 2, 3, 4}:
                logical_bytes = width * height * int(bpe)
                usable_data = raw_data[: (len(raw_data) // int(bpe)) * int(bpe)]
                base_data = usable_data[:logical_bytes]
                processed = base_data
                preview_width = width
                preview_height = height
                unsw_variant = "normal"
                if swizzle_verdict == "likely_swizzled_input":
                    try:
                        processed, preview_width, preview_height, unsw_variant, _ = (
                            _ps5_unswizzle_best_variant(
                                usable_data,
                                width,
                                height,
                                int(bpe),
                                allow_axis_swap=True,
                                roughness_guard=True,
                            )
                        )
                    except Exception:
                        processed = base_data
                        preview_width = width
                        preview_height = height
                        unsw_variant = "normal"
                mode_map = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}
                preview_image = Image.frombytes(
                    mode_map[int(bpe)],
                    (preview_width, preview_height),
                    processed,
                )
                if (
                    swizzle_verdict == "likely_swizzled_input"
                    and unsw_variant != "already_linear"
                ):
                    # KR: 축-스왑(전치)된 경우에만 회전 적용 (예: Alpha8, bpe=1)
                    # EN: Apply rotation only for axis-swapped (transposed) case (e.g. Alpha8, bpe=1)
                    if unsw_variant == "swapped_axes" and preview_rotate % 360 != 0:
                        preview_image = preview_image.rotate(
                            preview_rotate % 360, expand=True
                        )
                else:
                    # KR: linear(비-swizzle) 텍스쳐는 Unity 좌표계(Y=0 하단)로 저장되므로 상하 반전 보정
                    # EN: Linear (non-swizzle) textures are stored in Unity coordinates (Y=0 bottom), so flip vertically
                    preview_image = ImageOps.flip(preview_image)
                if unsw_variant == "addrlib_4KB_S":
                    # KR: addrlib 비압축 복원 경로는 Y축이 뒤집힌 사례(ui_button)가 있어 보정
                    # EN: Addrlib uncompressed restore path has cases with flipped Y-axis (ui_button), so correct it
                    preview_image = ImageOps.flip(preview_image)
                return preview_image

        image = getattr(texture, "image", None)
        if isinstance(image, Image.Image):
            preview_image = image
            if swizzle_verdict == "likely_swizzled_input":
                try:
                    preview_image = apply_ps5_unswizzle_to_image(
                        preview_image,
                        rotate=preview_rotate,
                        allow_axis_swap=True,
                        roughness_guard=True,
                    )
                except Exception:
                    pass
            return preview_image
    except Exception:
        return None
    return None


def _save_swizzle_preview(
    image: Image.Image,
    *,
    preview_enabled: bool,
    preview_root: str | None,
    assets_file_name: str,
    assets_name: str,
    atlas_path_id: int,
    font_name: str,
    target_swizzled: bool,
    lang: Language,
) -> None:
    """KR: swizzle 상태 확인용 preview 이미지를 PNG로 저장합니다.
    EN: Save a preview image for swizzle state verification as PNG.
    """
    if not (preview_enabled and preview_root):
        return
    try:
        visible = _preview_visible_image(image)
        file_dir = sanitize_filename_component(assets_file_name, fallback="assets_file")
        out_dir = os.path.join(preview_root, file_dir)
        os.makedirs(out_dir, exist_ok=True)
        safe_assets = sanitize_filename_component(assets_name, fallback="assets")
        safe_font = sanitize_filename_component(font_name, fallback="font")
        state_label = "target_swizzled" if target_swizzled else "target_linear"
        out_name = f"{safe_assets}__{atlas_path_id}__{safe_font}__unswizzled__{state_label}.png"
        out_path = os.path.join(out_dir, out_name)
        visible.save(out_path, format="PNG")
        if lang == "ko":
            _log_console(f"  Preview 저장: {out_path}")
        else:
            _log_console(f"  Preview saved: {out_path}")
    except Exception as preview_error:
        if lang == "ko":
            _log_console(f"  경고: preview 저장 실패 ({preview_error})")
        else:
            _log_console(f"  Warning: failed to save preview ({preview_error})")


def _save_glyph_crop_previews(
    image: Image.Image,
    *,
    preview_enabled: bool,
    preview_root: str | None,
    assets_file_name: str,
    assets_name: str,
    atlas_path_id: int,
    font_name: str,
    sdf_data: JsonDict,
    lang: Language,
) -> None:
    """KR: 글리프 테이블에서 개별 문자 crop preview를 PNG로 저장합니다.
    EN: Save individual character crop previews from the glyph table as PNG.
    """
    if not (preview_enabled and preview_root):
        return
    glyph_table = sdf_data.get("m_GlyphTable")
    char_table = sdf_data.get("m_CharacterTable")
    if not isinstance(glyph_table, list) or not isinstance(char_table, list):
        return
    try:
        visible = _preview_visible_image(image)
        file_dir = sanitize_filename_component(assets_file_name, fallback="assets_file")
        safe_assets = sanitize_filename_component(assets_name, fallback="assets")
        safe_font = sanitize_filename_component(font_name, fallback="font")
        glyph_dir = os.path.join(
            preview_root,
            file_dir,
            f"{safe_assets}__{atlas_path_id}__{safe_font}",
        )
        os.makedirs(glyph_dir, exist_ok=True)

        glyph_rect_by_index: dict[int, tuple[int, int, int, int]] = {}
        for glyph in glyph_table:
            if not isinstance(glyph, dict):
                continue
            try:
                glyph_index = int(glyph.get("m_Index", -1))
            except Exception:
                continue
            rect_raw = glyph.get("m_GlyphRect", {})
            if not isinstance(rect_raw, dict):
                continue
            try:
                gx = int(rect_raw.get("m_X", 0))
                gy = int(rect_raw.get("m_Y", 0))
                gw = int(rect_raw.get("m_Width", 0))
                gh = int(rect_raw.get("m_Height", 0))
            except Exception:
                continue
            if gw <= 0 or gh <= 0:
                continue
            glyph_rect_by_index[glyph_index] = (gx, gy, gw, gh)

        if not glyph_rect_by_index:
            return

        saved = 0
        used_names: set[str] = set()
        for ch in char_table:
            if not isinstance(ch, dict):
                continue
            try:
                codepoint = int(ch.get("m_Unicode", -1))
                glyph_index = int(ch.get("m_GlyphIndex", -1))
            except Exception:
                continue
            if codepoint < 0:
                continue
            rect = glyph_rect_by_index.get(glyph_index)
            if rect is None:
                continue

            x, y, w, h = rect
            # KR: TMP new GlyphRect.y는 bottom-origin이므로 PIL(top-origin) crop 좌표로 변환
            # EN: TMP new GlyphRect.y is bottom-origin, so convert to PIL (top-origin) crop coordinates
            y = int(round(_tmp_flip_y_between_old_new(y, h, visible.height)))
            x0 = max(0, min(visible.width, x))
            y0 = max(0, min(visible.height, y))
            x1 = max(0, min(visible.width, x + w))
            y1 = max(0, min(visible.height, y + h))
            if x1 <= x0 or y1 <= y0:
                continue

            base = f"U+{codepoint:04X}"
            try:
                ch_text = chr(codepoint)
                if ch_text.isprintable() and not ch_text.isspace():
                    safe_char = sanitize_filename_component(
                        ch_text, fallback="", max_len=8
                    )
                    if safe_char and safe_char != "unnamed":
                        base = f"{base}_{safe_char}"
            except Exception:
                pass

            name = base
            if name in used_names:
                name = f"{name}_g{glyph_index}"
            used_names.add(name)
            out_path = os.path.join(glyph_dir, f"{name}.png")
            visible.crop((x0, y0, x1, y1)).save(out_path, format="PNG")
            saved += 1

        if saved > 0:
            if lang == "ko":
                _log_console(f"  Glyph preview 저장: {saved}개 -> {glyph_dir}")
            else:
                _log_console(f"  Glyph previews saved: {saved} -> {glyph_dir}")
    except Exception as preview_error:
        if lang == "ko":
            _log_console(f"  경고: glyph preview 저장 실패 ({preview_error})")
        else:
            _log_console(f"  Warning: failed to save glyph previews ({preview_error})")


def _prepare_texture_replacement_for_target(
    texture_plan: JsonDict,
    *,
    assets_file_name: str,
    target_assets_name: str,
    target_path_id: int,
    texture_object_lookup: dict[tuple[str, int], Any],
    texture_swizzle_state_cache: dict[str, tuple[str | None, str | None]],
    ps5_swizzle: bool,
    preview_export: bool,
    preview_root: str | None,
    lang: Language,
) -> JsonDict | None:
    """KR: 교체 Atlas의 swizzle 상태를 타겟에 맞추고 preview를 생성합니다.
    EN: Match the replacement atlas swizzle state to the target and generate previews.
    """
    source_atlas = _load_spilled_plan_image(
        texture_plan,
        image_key="source_atlas",
        path_key="source_atlas_path",
    )
    if not isinstance(source_atlas, Image.Image):
        return None
    owned_images: list[Image.Image] = []
    if not isinstance(texture_plan.get("source_atlas"), Image.Image):
        owned_images.append(source_atlas)

    source_path = str(texture_plan.get("source_atlas_path", "")).strip()
    alpha_path = str(texture_plan.get("alpha8_linear_source_path", "")).strip()
    if (
        source_path
        and alpha_path
        and os.path.normcase(os.path.abspath(source_path))
        == os.path.normcase(os.path.abspath(alpha_path))
    ):
        alpha8_linear_source = source_atlas
    else:
        alpha8_linear_source = _load_spilled_plan_image(
            texture_plan,
            image_key="alpha8_linear_source",
            path_key="alpha8_linear_source_path",
        )
        if (
            isinstance(alpha8_linear_source, Image.Image)
            and alpha8_linear_source is not source_atlas
            and not isinstance(texture_plan.get("alpha8_linear_source"), Image.Image)
        ):
            owned_images.append(alpha8_linear_source)
    atlas_linear_for_alpha8 = (
        alpha8_linear_source
        if isinstance(alpha8_linear_source, Image.Image)
        else source_atlas
    )
    source_swizzled = parse_bool_flag(texture_plan.get("source_swizzled"))
    replacement_swizzle_hint = parse_bool_flag(
        texture_plan.get("replacement_swizzle_hint")
    )
    replacement_process_swizzle = parse_bool_flag(
        texture_plan.get("replacement_process_swizzle")
    )
    asset_process_swizzle = parse_bool_flag(texture_plan.get("asset_process_swizzle"))
    font_name = str(
        texture_plan.get("font_name")
        or texture_plan.get("replacement_font")
        or f"Texture_{target_path_id}"
    )
    try:
        atlas_metadata_width = int(
            texture_plan.get("metadata_width", source_atlas.width) or source_atlas.width
        )
    except Exception:
        atlas_metadata_width = int(source_atlas.width)
    try:
        atlas_metadata_height = int(
            texture_plan.get("metadata_height", source_atlas.height)
            or source_atlas.height
        )
    except Exception:
        atlas_metadata_height = int(source_atlas.height)

    target_swizzle_verdict: str | None = None
    target_swizzle_source: str | None = None
    target_is_swizzled: bool | None = None
    desired_swizzle_state = source_swizzled

    if ps5_swizzle:
        target_swizzle_verdict, target_swizzle_source = _detect_target_texture_swizzle(
            texture_object_lookup,
            texture_swizzle_state_cache,
            target_assets_name,
            int(target_path_id),
        )
        if target_swizzle_verdict == "likely_swizzled_input":
            target_is_swizzled = True
        elif target_swizzle_verdict == "likely_linear_input":
            target_is_swizzled = False
        elif replacement_swizzle_hint:
            target_is_swizzled = True

        if target_is_swizzled is not None:
            desired_swizzle_state = target_is_swizzled

    if replacement_process_swizzle or asset_process_swizzle:
        desired_swizzle_state = True

    if ps5_swizzle:
        if target_swizzle_verdict == "likely_swizzled_input":
            reason = (
                f" (근거: {target_swizzle_source})"
                if lang == "ko" and target_swizzle_source
                else (
                    f" (source: {target_swizzle_source})"
                    if target_swizzle_source
                    else ""
                )
            )
            if lang == "ko":
                _log_console(
                    f"  PS5 swizzle 감지: 대상 Atlas가 swizzled 상태로 판별되었습니다.{reason}"
                )
            else:
                _log_console(
                    f"  PS5 swizzle detect: target atlas is likely swizzled.{reason}"
                )
        elif target_swizzle_verdict == "likely_linear_input":
            reason = (
                f" (근거: {target_swizzle_source})"
                if lang == "ko" and target_swizzle_source
                else (
                    f" (source: {target_swizzle_source})"
                    if target_swizzle_source
                    else ""
                )
            )
            if lang == "ko":
                _log_console(
                    f"  PS5 swizzle 감지: 대상 Atlas가 선형(linear) 상태로 판별되었습니다.{reason}"
                )
            else:
                _log_console(
                    f"  PS5 swizzle detect: target atlas is likely linear.{reason}"
                )
        elif replacement_swizzle_hint:
            if lang == "ko":
                _log_console(
                    "  PS5 swizzle 힌트: JSON swizzle=yes 값을 기준으로 swizzle 적용합니다."
                )
            else:
                _log_console(
                    "  PS5 swizzle hint: applying swizzle based on JSON swizzle=yes."
                )
        elif lang == "ko":
            _log_console(
                "  PS5 swizzle 감지: inconclusive, 교체 Atlas 원본 상태를 유지합니다."
            )
        else:
            _log_console(
                "  PS5 swizzle detect: inconclusive, keeping replacement atlas state."
            )
    elif replacement_process_swizzle:
        if lang == "ko":
            _log_console(
                "  process_swizzle=True: 교체 Atlas를 swizzle 상태로 변환합니다."
            )
        else:
            _log_console(
                "  process_swizzle=True: converting replacement atlas to swizzled state."
            )

    _log_debug(
        f"[replace_texture_plan] file={assets_file_name} assets={target_assets_name} "
        f"path_id={target_path_id} source_swizzled={source_swizzled} "
        f"target_swizzle_verdict={target_swizzle_verdict} "
        f"target_swizzle_source={target_swizzle_source} "
        f"desired_swizzle={desired_swizzle_state}"
    )

    atlas_for_write = source_atlas
    if desired_swizzle_state != source_swizzled:
        try:
            if desired_swizzle_state:
                atlas_for_write = apply_ps5_swizzle_to_image(source_atlas)
            else:
                atlas_for_write = apply_ps5_unswizzle_to_image(source_atlas)
            if atlas_for_write is not source_atlas:
                owned_images.append(atlas_for_write)
        except Exception as swizzle_error:
            atlas_for_write = source_atlas
            if lang == "ko":
                _log_console(
                    f"  경고: PS5 swizzle 변환 실패, 원본 Atlas를 사용합니다. ({swizzle_error})"
                )
            else:
                _log_console(
                    f"  Warning: PS5 swizzle transform failed; using original atlas. ({swizzle_error})"
                )

    if preview_export:
        preview_image = atlas_for_write
        if ps5_swizzle and desired_swizzle_state:
            try:
                preview_image = apply_ps5_unswizzle_to_image(atlas_for_write)
                if preview_image is not atlas_for_write:
                    owned_images.append(preview_image)
            except Exception as preview_unswizzle_error:
                preview_image = atlas_for_write
                if lang == "ko":
                    _log_console(
                        "  경고: preview unswizzle 실패, 저장 상태 Atlas 그대로 미리보기를 저장합니다. "
                        f"({preview_unswizzle_error})"
                    )
                else:
                    _log_console(
                        "  Warning: preview unswizzle failed; saving preview from stored atlas state. "
                        f"({preview_unswizzle_error})"
                    )
        _save_swizzle_preview(
            preview_image,
            preview_enabled=preview_export,
            preview_root=preview_root,
            assets_file_name=assets_file_name,
            assets_name=target_assets_name,
            atlas_path_id=int(target_path_id),
            font_name=font_name,
            target_swizzled=bool(desired_swizzle_state),
            lang=lang,
        )
        preview_sdf_data = texture_plan.get("preview_sdf_data")
        if isinstance(preview_sdf_data, dict):
            _save_glyph_crop_previews(
                preview_image,
                preview_enabled=preview_export,
                preview_root=preview_root,
                assets_file_name=assets_file_name,
                assets_name=target_assets_name,
                atlas_path_id=int(target_path_id),
                font_name=font_name,
                sdf_data=preview_sdf_data,
                lang=lang,
            )

    return {
        "replacement_image": atlas_for_write,
        # KR: Alpha8 raw 저장에는 자동 판정값이 아니라 강제 옵션까지 반영된
        #     최종 저장 상태가 필요합니다.
        # EN: Alpha8 raw encoding needs the final storage state, including
        #     process_swizzle overrides, rather than only the detector verdict.
        "target_swizzled_state": bool(desired_swizzle_state),
        "replacement_linear_source": atlas_linear_for_alpha8,
        "metadata_size": (
            int(atlas_metadata_width),
            int(atlas_metadata_height),
        ),
        "_owned_images": owned_images,
    }


def _image_to_alpha8_bytes(image: Image.Image) -> tuple[bytes, int, int]:
    """KR: Pillow 이미지를 Alpha8 raw bytes로 변환합니다.
    EN: Convert a Pillow image to Alpha8 raw bytes.
    """
    if image.mode in {"RGBA", "LA"}:
        alpha = image.getchannel("A")
    elif image.mode == "L":
        alpha = image
    else:
        alpha = image.convert("L")
    return alpha.tobytes(), alpha.width, alpha.height


def _encode_alpha8_replacement_bytes(
    alpha_source: Image.Image,
    *,
    ps5_swizzle: bool,
    target_swizzled_state: bool | None,
) -> tuple[bytes, int, int, str]:
    """KR: Alpha8 교체 바이트를 타겟 swizzle 상태에 맞게 인코딩합니다.
    EN: Encode Alpha8 replacement bytes to match the target swizzle state.
    """
    if ps5_swizzle and target_swizzled_state is True:
        alpha_linear, aw, ah = _image_to_alpha8_bytes(alpha_source)
        alpha_linear_img = Image.frombytes("L", (int(aw), int(ah)), alpha_linear)
        alpha_swizzled_img = apply_ps5_swizzle_to_image(alpha_linear_img)
        alpha_raw, aw, ah = _image_to_alpha8_bytes(alpha_swizzled_img)
        return alpha_raw, aw, ah, "swizzled"

    if (not ps5_swizzle) or target_swizzled_state is False:
        alpha_raw, aw, ah = _image_to_alpha8_bytes(ImageOps.flip(alpha_source))
        return alpha_raw, aw, ah, "linear_flipped"

    alpha_raw, aw, ah = _image_to_alpha8_bytes(alpha_source)
    return alpha_raw, aw, ah, "direct"


def build_replacement_lookup(
    replacements: dict[str, JsonDict],
) -> tuple[dict[tuple[str, str, str, int], str], set[str]]:
    """KR: 교체 JSON을 빠른 조회용 룩업 테이블로 변환합니다.
    (Type, File, assets_name, Path_ID) → font_name 매핑을 생성합니다.
    EN: Converts the replacement JSON into a fast-lookup table.
    Builds a (Type, File, assets_name, Path_ID) → font_name mapping.
    """
    lookup: dict[tuple[str, str, str, int], str] = {}
    files_to_process: set[str] = set()

    for info in replacements.values():
        replace_to = info.get("Replace_to")
        if not replace_to:
            continue

        file_name_raw = info.get("File")
        assets_name_raw = info.get("assets_name")
        path_id_raw = info.get("Path_ID")
        type_name_raw = info.get("Type")

        if not isinstance(file_name_raw, str) or not file_name_raw:
            continue
        if not isinstance(assets_name_raw, str) or not assets_name_raw:
            continue
        if not isinstance(type_name_raw, str) or not type_name_raw:
            continue
        if path_id_raw is None:
            continue

        try:
            path_id = int(path_id_raw)
        except (TypeError, ValueError):
            continue

        lookup[(type_name_raw, file_name_raw, assets_name_raw, path_id)] = str(
            replace_to
        ).strip()
        files_to_process.add(file_name_raw)

    return lookup, files_to_process


def load_replacement_mapping_file(json_file: str) -> dict[str, JsonDict]:
    with open(json_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError("JSON root must be an object (dict).")
    fonts = loaded.get("fonts")
    if isinstance(fonts, dict):
        return cast(dict[str, JsonDict], fonts)
    return cast(dict[str, JsonDict], loaded)


def debug_parse_enabled() -> bool:
    """KR: 디버그 파싱 로그 활성화 여부를 반환합니다.
    EN: Returns whether debug parse logging is enabled.
    """
    return os.environ.get("UFR_DEBUG_PARSE", "").strip() == "1"


def debug_parse_log(message: str) -> None:
    """KR: 디버그 모드일 때만 파싱 로그를 출력합니다.
    EN: Outputs parse logs only when debug mode is active.
    """
    if debug_parse_enabled():
        _log_console(message)


def _log_scan_result_details(
    file_name: str, scanned: dict[str, list[JsonDict]]
) -> None:
    """KR: 스캔 결과를 파일/폰트 단위 DEBUG 로그로 남깁니다.
    EN: Logs scan results at file/font level as DEBUG output.
    """
    ttf_entries = list(scanned.get("ttf", []))
    sdf_entries = list(scanned.get("sdf", []))
    _log_debug(
        f"[scan_debug] file={file_name} ttf_count={len(ttf_entries)} sdf_count={len(sdf_entries)}"
    )

    for font_entry in ttf_entries:
        assets_name = str(font_entry.get("assets_name", ""))
        font_name = str(font_entry.get("name", ""))
        path_id = font_entry.get("path_id")
        _log_debug(
            f"[scan_debug] type=TTF file={file_name} assets={assets_name} path_id={path_id} name={font_name}"
        )

    for font_entry in sdf_entries:
        assets_name = str(font_entry.get("assets_name", ""))
        font_name = str(font_entry.get("name", ""))
        path_id = font_entry.get("path_id")
        swizzle = font_entry.get("swizzle")
        swizzle_text = f" swizzle={swizzle}" if swizzle is not None else ""
        _log_debug(
            f"[scan_debug] type=SDF file={file_name} assets={assets_name} path_id={path_id} name={font_name}{swizzle_text}"
        )


def _log_replacement_plan_details(
    file_name: str,
    replacement_mapping: dict[str, JsonDict],
) -> None:
    """KR: 파일별 교체 계획을 DEBUG 로그로 기록합니다.
    EN: Records the per-file replacement plan as DEBUG log.
    """
    if not replacement_mapping:
        _log_debug(f"[replace_plan] file={file_name} targets=0")
        return

    ttf_count = sum(
        1 for item in replacement_mapping.values() if item.get("Type") == "TTF"
    )
    sdf_count = sum(
        1 for item in replacement_mapping.values() if item.get("Type") == "SDF"
    )
    _log_debug(
        f"[replace_plan] file={file_name} targets={len(replacement_mapping)} ttf={ttf_count} sdf={sdf_count}"
    )

    for entry_key in sorted(replacement_mapping.keys()):
        entry = replacement_mapping[entry_key]
        type_name = str(entry.get("Type", ""))
        assets_name = str(entry.get("assets_name", ""))
        path_id = entry.get("Path_ID")
        source_name = str(entry.get("Name", ""))
        replace_to = str(entry.get("Replace_to", ""))
        force_raster = entry.get("force_raster")
        swizzle = entry.get("swizzle")
        process_swizzle = entry.get("process_swizzle")
        extra_flags = ""
        if (
            force_raster is not None
            or swizzle is not None
            or process_swizzle is not None
        ):
            extra_flags = (
                f" force_raster={force_raster} swizzle={swizzle} "
                f"process_swizzle={process_swizzle}"
            )
        _log_debug(
            f"[replace_plan] type={type_name} file={file_name} assets={assets_name} path_id={path_id} "
            f"name={source_name} replace_to={replace_to}{extra_flags}"
        )


def _normalize_assets_basename(value: Any) -> str | None:
    """KR: 에셋 경로에서 파일명(basename)만 정규화하여 반환합니다.
    EN: Normalizes and returns only the filename (basename) from an asset path.
    """
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    normalized = text.replace("\\", "/")
    name = os.path.basename(normalized)
    return name or None


def _normalize_asset_lookup_path(value: Any) -> str | None:
    """KR: 에셋 조회용 경로를 정규화합니다. archive://, file:// 접두사를 제거하고 소문자로 변환합니다.
    EN: Normalizes the asset lookup path. Strips archive://, file:// prefixes and converts to lowercase.
    """
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    normalized = text.replace("\\", "/")
    lowered = normalized.lower()
    for prefix in ("archive://", "archive:/", "file://"):
        if lowered.startswith(prefix):
            normalized = normalized[len(prefix) :]
            lowered = normalized.lower()
            break
    while normalized.startswith("/"):
        normalized = normalized[1:]
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = re.sub(r"/{2,}", "/", normalized).strip()
    return normalized.lower() if normalized else None


def _normalize_asset_file_key(path: Any) -> str | None:
    """KR: 에셋 파일 경로를 절대경로 기반의 정규화된 키로 변환합니다.
    EN: Converts an asset file path to a normalized key based on its absolute path.
    """
    text = str(path).strip() if path is not None else ""
    if not text:
        return None
    return os.path.normcase(os.path.abspath(text))


def _build_asset_file_index(
    all_assets_files: list[str],
    data_path: str,
) -> dict[str, Any]:
    """KR: 모든 에셋 파일 목록으로부터 상대경로/basename 기반 인덱스를 구축합니다.
    EN: Builds a relative-path/basename-based index from the full list of asset files.
    """
    data_root = os.path.abspath(data_path)
    path_by_key: dict[str, str] = {}
    relpath_to_keys: dict[str, list[str]] = {}
    basename_to_keys: dict[str, list[str]] = {}
    relpath_by_key: dict[str, str] = {}
    basename_by_key: dict[str, str] = {}

    for candidate_path in sorted(all_assets_files):
        key = _normalize_asset_file_key(candidate_path)
        if not key:
            continue
        abs_path = os.path.abspath(candidate_path)
        rel_path = os.path.relpath(abs_path, data_root).replace("\\", "/").lower()
        basename = os.path.basename(abs_path).lower()
        path_by_key[key] = abs_path
        relpath_by_key[key] = rel_path
        basename_by_key[key] = basename
        relpath_to_keys.setdefault(rel_path, []).append(key)
        basename_to_keys.setdefault(basename, []).append(key)

    return {
        "data_root": data_root,
        "path_by_key": path_by_key,
        "relpath_to_keys": relpath_to_keys,
        "basename_to_keys": basename_to_keys,
        "relpath_by_key": relpath_by_key,
        "basename_by_key": basename_by_key,
    }


def _extract_external_assets_name(external_ref: Any) -> str | None:
    """KR: 외부 참조 객체에서 에셋 이름(basename)을 추출합니다.
    EN: Extracts the asset name (basename) from an external reference object.
    """
    if external_ref is None:
        return None

    candidates: list[Any] = []
    if isinstance(external_ref, dict):
        candidates.extend(
            [
                external_ref.get("path"),
                external_ref.get("pathName"),
                external_ref.get("name"),
                external_ref.get("fileName"),
                external_ref.get("asset_name"),
                external_ref.get("assetPath"),
            ]
        )
    else:
        for attr in (
            "path",
            "pathName",
            "name",
            "fileName",
            "asset_name",
            "assetPath",
        ):
            candidates.append(getattr(external_ref, attr, None))

    for candidate in candidates:
        name = _normalize_assets_basename(candidate)
        if name:
            return name
    return None


def _extract_external_assets_candidates(external_ref: Any) -> list[str]:
    """KR: 외부 참조 객체에서 가능한 모든 에셋 경로/이름 후보를 추출합니다.
    EN: Extracts all possible asset path/name candidates from an external reference object.
    """
    if external_ref is None:
        return []

    raw_candidates: list[Any] = []
    if isinstance(external_ref, dict):
        raw_candidates.extend(
            [
                external_ref.get("path"),
                external_ref.get("pathName"),
                external_ref.get("name"),
                external_ref.get("fileName"),
                external_ref.get("asset_name"),
                external_ref.get("assetPath"),
            ]
        )
    else:
        for attr in (
            "path",
            "pathName",
            "name",
            "fileName",
            "asset_name",
            "assetPath",
        ):
            raw_candidates.append(getattr(external_ref, attr, None))

    resolved: list[str] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        normalized_path = _normalize_asset_lookup_path(candidate)
        if normalized_path and normalized_path not in seen:
            seen.add(normalized_path)
            resolved.append(normalized_path)
        normalized_name = _normalize_assets_basename(candidate)
        if normalized_name:
            lowered_name = normalized_name.lower()
            if lowered_name not in seen:
                seen.add(lowered_name)
                resolved.append(lowered_name)
    return resolved


def _resolve_external_ref(source_assets_file: Any, file_id: int) -> Any:
    """KR: FileID를 사용하여 소스 에셋 파일의 externals 목록에서 외부 참조를 조회합니다. FileID=0은 같은 파일, FileID>0은 externals 리스트의 1-based 인덱스입니다.
    EN: Looks up an external reference from the source asset file's externals list using FileID. FileID=0 means the same file; FileID>0 is a 1-based index into the externals list.
    """
    try:
        resolved_file_id = int(file_id or 0)
    except Exception:
        resolved_file_id = 0

    if resolved_file_id == 0:
        return None

    externals = getattr(source_assets_file, "externals", None)
    if externals is None:
        externals = getattr(source_assets_file, "m_Externals", None)

    if isinstance(externals, dict):
        external_ref = externals.get(resolved_file_id)
        if external_ref is None:
            external_ref = externals.get(resolved_file_id - 1)
        return external_ref

    if isinstance(externals, (list, tuple)):
        ext_index = resolved_file_id - 1
        if 0 <= ext_index < len(externals):
            return externals[ext_index]
    return None


def _resolve_assets_name_from_file_id(source_assets_file: Any, file_id: int) -> str | None:
    """KR: FileID로부터 대상 에셋 파일 이름을 확인합니다. FileID=0이면 현재 파일 이름을 반환합니다.
    EN: Resolves the target asset file name from a FileID. Returns the current file name if FileID=0.
    """
    try:
        resolved_file_id = int(file_id or 0)
    except Exception:
        resolved_file_id = 0

    if resolved_file_id == 0:
        return _normalize_assets_basename(getattr(source_assets_file, "name", ""))

    externals = getattr(source_assets_file, "externals", None)
    if externals is None:
        externals = getattr(source_assets_file, "m_Externals", None)

    external_ref = _resolve_external_ref(source_assets_file, resolved_file_id)
    if externals is None:
        return None
    return _extract_external_assets_name(external_ref)


def _resolve_target_assets_name(
    source_assets_file: Any,
    current_assets_name: str,
    file_id: int,
) -> str | None:
    """KR: FileID 기반으로 대상 에셋 이름을 결정합니다. FileID=0이면 현재 에셋 이름을 그대로 반환합니다.
    EN: Determines the target asset name based on FileID. Returns the current asset name as-is if FileID=0.
    """
    try:
        resolved_file_id = int(file_id or 0)
    except Exception:
        resolved_file_id = 0
    if resolved_file_id == 0:
        return str(current_assets_name)
    return _resolve_assets_name_from_file_id(source_assets_file, resolved_file_id)


def _resolve_material_main_texture_key(
    source_assets_file: Any,
    current_assets_name: str,
    material: Any,
) -> str | None:
    """Resolve a Material's _MainTex reference across asset files."""
    saved_props = getattr(material, "m_SavedProperties", None)
    tex_envs = getattr(saved_props, "m_TexEnvs", None)
    if not isinstance(tex_envs, list):
        return None
    for entry in tex_envs:
        if not (
            isinstance(entry, (list, tuple))
            and len(entry) >= 2
            and str(entry[0]) == "_MainTex"
        ):
            continue
        tex_env = entry[1]
        tex_ref = (
            tex_env.get("m_Texture")
            if isinstance(tex_env, dict)
            else getattr(tex_env, "m_Texture", None)
        )
        if isinstance(tex_ref, dict):
            file_id = int(tex_ref.get("m_FileID", 0) or 0)
            path_id = int(tex_ref.get("m_PathID", 0) or 0)
        else:
            file_id = int(getattr(tex_ref, "m_FileID", 0) or 0)
            path_id = int(getattr(tex_ref, "m_PathID", 0) or 0)
        if path_id <= 0:
            return None
        target_assets_name = _resolve_target_assets_name(
            source_assets_file,
            current_assets_name,
            file_id,
        )
        if not target_assets_name:
            return None
        return _make_assets_object_key(target_assets_name, path_id)
    return None


def _collect_asset_file_index_matches(
    asset_file_index: dict[str, Any] | None,
    reference: Any,
) -> list[str]:
    """KR: 에셋 파일 인덱스에서 참조 문자열과 일치하는 모든 키를 수집합니다.
    EN: Collects all keys from the asset file index that match the reference string.
    """
    if not isinstance(asset_file_index, dict):
        return []

    normalized_reference = _normalize_asset_lookup_path(reference)
    if not normalized_reference:
        normalized_reference = _normalize_assets_basename(reference)
        if normalized_reference:
            normalized_reference = normalized_reference.lower()
    if not normalized_reference:
        return []

    relpath_to_keys = cast(
        dict[str, list[str]],
        asset_file_index.get("relpath_to_keys", {}),
    )
    basename_to_keys = cast(
        dict[str, list[str]],
        asset_file_index.get("basename_to_keys", {}),
    )
    relpath_by_key = cast(dict[str, str], asset_file_index.get("relpath_by_key", {}))

    matches: list[str] = []
    seen: set[str] = set()

    def _append_match(match_key: str) -> None:
        if match_key and match_key not in seen:
            seen.add(match_key)
            matches.append(match_key)

    for match_key in relpath_to_keys.get(normalized_reference, []):
        _append_match(match_key)

    if not matches and "/" in normalized_reference:
        suffix = "/" + normalized_reference
        for match_key, rel_path in relpath_by_key.items():
            if rel_path == normalized_reference or rel_path.endswith(suffix):
                _append_match(match_key)

    basename = os.path.basename(normalized_reference)
    for match_key in basename_to_keys.get(basename, []):
        _append_match(match_key)

    return matches


def _choose_asset_file_match(
    asset_file_index: dict[str, Any] | None,
    matches: list[str],
    *,
    current_file_key: str | None,
    reference_desc: str,
) -> str | None:
    """KR: 여러 일치 항목 중 하나를 선택합니다. 같은 디렉토리의 형제 파일을 우선하고, 모호하면 정렬 후 첫 번째를 사용합니다.
    EN: Selects one match from multiple candidates. Prefers sibling files in the same directory; if ambiguous, sorts and uses the first.
    """
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    if current_file_key and isinstance(asset_file_index, dict):
        path_by_key = cast(dict[str, str], asset_file_index.get("path_by_key", {}))
        current_path = path_by_key.get(current_file_key)
        if current_path:
            current_dir = os.path.dirname(current_path)
            sibling_matches = [
                match_key
                for match_key in matches
                if os.path.dirname(path_by_key.get(match_key, "")) == current_dir
            ]
            if len(sibling_matches) == 1:
                return sibling_matches[0]
    chosen = sorted(matches)[0]
    _log_warning(
        f"[asset_path_ambiguous] reference={reference_desc} match_count={len(matches)} "
        f"using_first={chosen}"
    )
    return chosen


def _resolve_target_outer_file_key(
    current_file_key: str,
    source_assets_file: Any,
    file_id: int,
    target_assets_name: str | None,
    *,
    source_bundle_signature: str | None,
    asset_file_index: dict[str, Any] | None,
) -> str | None:
    """KR: FileID와 에셋 이름으로 실제 대상 outer 파일 키를 확인합니다.
    EN: Resolve the actual target outer-file key from FileID and asset name.
    """
    try:
        resolved_file_id = int(file_id or 0)
    except Exception:
        resolved_file_id = 0
    if resolved_file_id == 0:
        return str(current_file_key)

    external_ref = _resolve_external_ref(source_assets_file, resolved_file_id)
    candidates = _extract_external_assets_candidates(external_ref)
    if target_assets_name:
        normalized_assets_name = _normalize_assets_basename(target_assets_name)
        if normalized_assets_name:
            candidates.append(normalized_assets_name.lower())

    if source_bundle_signature in BUNDLE_SIGNATURES:
        environment = getattr(source_assets_file, "environment", None)
        roots = getattr(environment, "files", None)
        stack = list(roots.values()) if isinstance(roots, dict) else []
        internal_names: set[str] = set()
        seen: set[int] = set()
        while stack:
            item = stack.pop()
            if item is None or id(item) in seen:
                continue
            seen.add(id(item))
            item_name = _normalize_assets_basename(getattr(item, "name", None))
            if item_name:
                internal_names.add(item_name.lower())
            children = getattr(item, "files", None)
            if isinstance(children, dict):
                for child_name, child in children.items():
                    normalized_child = _normalize_assets_basename(child_name)
                    if normalized_child:
                        internal_names.add(normalized_child.lower())
                    stack.append(child)
        if any(
            _normalize_assets_basename(candidate).lower() in internal_names
            for candidate in candidates
            if _normalize_assets_basename(candidate)
        ):
            return str(current_file_key)

    for candidate in candidates:
        matches = _collect_asset_file_index_matches(asset_file_index, candidate)
        chosen = _choose_asset_file_match(
            asset_file_index,
            matches,
            current_file_key=current_file_key,
            reference_desc=str(candidate),
        )
        if chosen:
            return chosen
    return None


def _make_assets_object_key(assets_name: str, path_id: int) -> str:
    """KR: 에셋 이름과 PathID를 결합하여 고유 객체 키 문자열을 생성합니다.
    EN: Creates a unique object key string by combining the asset name and PathID.
    """
    return f"{str(assets_name)}|{int(path_id)}"


def _lookup_patch_value(mapping: dict[str, Any], key: str) -> Any | None:
    """KR: 패치 맵에서 키를 조회합니다. 대소문자 구분 후 소문자 폴백을 시도합니다.
    EN: Looks up a key in the patch map. Tries case-sensitive first, then falls back to lowercase.
    """
    if key in mapping:
        return mapping[key]
    lowered = key.lower()
    if lowered in mapping:
        return mapping[lowered]
    return None


def _store_patch_value(mapping: dict[str, Any], key: str, value: Any) -> None:
    """KR: 패치 맵에 값을 저장합니다. 원본 키와 소문자 키 양쪽에 동시 저장합니다.
    EN: Stores a value in the patch map. Saves to both the original key and its lowercase variant.
    """
    lowered = key.lower()
    for existing_key in list(mapping):
        if str(existing_key).lower() == lowered:
            mapping[existing_key] = value
    mapping[key] = value
    mapping[lowered] = value


def _store_consistent_patch_value(
    mapping: dict[str, Any],
    key: str,
    value: Any,
    *,
    patch_kind: str,
    target_file_key: str,
    transaction: _DeferredPatchTransaction | None,
) -> tuple[Any | None, bool]:
    """Store a consistent plan and report whether this call inserted it."""
    existing = _lookup_patch_value(mapping, key)
    if existing is not None and _deferred_patch_fingerprint(
        patch_kind, existing
    ) != _deferred_patch_fingerprint(patch_kind, value):
        conflict = (
            f"kind={patch_kind} file={target_file_key} key={key} "
            "contains incompatible payloads"
        )
        _log_warning(f"[patch_plan_conflict] {conflict}")
        if transaction is not None:
            transaction.fail(conflict)
        return None, False
    retained_value = existing if existing is not None else value
    if transaction is not None and not transaction.register_plan(
        patch_kind,
        target_file_key,
        key,
        retained_value,
    ):
        return None, False
    if existing is not None and existing is not value:
        _cleanup_superseded_patch_payload(value, existing)
    _store_patch_value(mapping, key, retained_value)
    return retained_value, existing is None


def _copy_patch_bucket(
    patch_map: dict[str, dict[str, Any]] | None,
    file_key: str,
) -> dict[str, Any]:
    """KR: 패치 맵에서 파일 키에 해당하는 버킷을 복사하여 반환합니다.
    EN: Copies and returns the bucket corresponding to the file key from the patch map.
    """
    if not isinstance(patch_map, dict):
        return {}
    bucket = patch_map.get(str(file_key), {})
    return dict(bucket) if isinstance(bucket, dict) else {}


def _build_material_atlas_reconciliation_buckets(
    file_keys: Iterable[str],
    material_atlas_plans: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Broadcast collected atlas fallbacks for a final material-only pass."""
    buckets: dict[str, dict[str, Any]] = {}
    if not material_atlas_plans:
        return buckets
    for file_key in file_keys:
        normalized_file = _normalize_asset_file_key(file_key)
        if normalized_file and normalized_file not in buckets:
            buckets[normalized_file] = dict(material_atlas_plans)
    return buckets


def _patch_payload_ids(bucket: dict[str, Any] | None) -> set[int]:
    """Return unique payload identities from a bucket containing alias keys."""
    if not isinstance(bucket, dict):
        return set()
    return {id(payload) for payload in bucket.values()}


def _consume_deferred_patch_payloads(
    patch_map: dict[str, dict[str, Any]] | None,
    file_key: str,
    consumed_payload_ids: set[int],
) -> int:
    """Remove only payloads that were successfully handled for one target file."""
    if not (isinstance(patch_map, dict) and consumed_payload_ids):
        return 0
    normalized_file = _normalize_asset_file_key(file_key) or str(file_key)
    bucket = patch_map.get(normalized_file)
    if not isinstance(bucket, dict):
        return 0

    removed: dict[str, Any] = {}
    removed_payload_ids: set[int] = set()
    for object_key, payload in list(bucket.items()):
        payload_id = id(payload)
        if payload_id not in consumed_payload_ids:
            continue
        removed[object_key] = payload
        removed_payload_ids.add(payload_id)
        del bucket[object_key]
    if not bucket:
        patch_map.pop(normalized_file, None)
    _cleanup_deferred_patch_bucket(removed)
    return len(removed_payload_ids)


def _resolve_current_file_key(assets_file: str, logical_file_key: str | None) -> str:
    """Keep deferred-patch identity separate from an output-only copy path."""
    return (
        _normalize_asset_file_key(logical_file_key) if logical_file_key else None
    ) or _normalize_asset_file_key(assets_file) or os.path.abspath(assets_file)


def _spill_image_to_temp_file(
    image: Image.Image,
    deferred_dir: str,
    *,
    prefix: str,
) -> str:
    """KR: PIL 이미지를 임시 PNG 파일로 저장하고 경로를 반환합니다. 메모리 절감을 위한 디스크 스필 처리입니다.
    EN: Saves a PIL image to a temporary PNG file and returns the path. This is a disk spill operation for memory savings.
    """
    os.makedirs(deferred_dir, exist_ok=True)
    fd, spill_path = tempfile.mkstemp(
        prefix=prefix,
        suffix=".png",
        dir=deferred_dir,
    )
    os.close(fd)
    image.save(spill_path, format="PNG")
    return spill_path


def _close_unique_images(*images: Any) -> None:
    seen: set[int] = set()
    for image in images:
        if not isinstance(image, Image.Image) or id(image) in seen:
            continue
        seen.add(id(image))
        try:
            image.close()
        except Exception:
            pass


def _spill_deferred_texture_plan_to_disk(
    texture_plan: JsonDict,
    deferred_dir: str,
) -> JsonDict:
    """KR: 지연 텍스처 계획의 이미지 데이터를 디스크 임시 파일로 스필합니다. 2패스 메커니즘에서 메모리를 절감합니다.
    EN: Spills image data from a deferred texture plan to temporary disk files. Saves memory in the 2-pass mechanism.
    """
    source_atlas = texture_plan.get("source_atlas")
    if not isinstance(source_atlas, Image.Image):
        return texture_plan

    spilled_plan = dict(texture_plan)
    atlas_path = str(spilled_plan.get("source_atlas_path", "")).strip()
    if not atlas_path:
        atlas_path = _spill_image_to_temp_file(
            source_atlas,
            deferred_dir,
            prefix="atlas_",
        )
    spilled_plan.pop("source_atlas", None)
    spilled_plan["source_atlas_path"] = atlas_path

    alpha_image = spilled_plan.get("alpha8_linear_source")
    if isinstance(alpha_image, Image.Image):
        alpha_path = str(spilled_plan.get("alpha8_linear_source_path", "")).strip()
        if not alpha_path:
            if alpha_image is source_atlas:
                alpha_path = atlas_path
            else:
                alpha_path = _spill_image_to_temp_file(
                    alpha_image,
                    deferred_dir,
                    prefix="alpha8_",
                )
        spilled_plan.pop("alpha8_linear_source", None)
        spilled_plan["alpha8_linear_source_path"] = alpha_path
    return spilled_plan


def _load_spilled_plan_image(
    payload: JsonDict,
    *,
    image_key: str,
    path_key: str,
) -> Image.Image | None:
    """KR: 디스크에 스필된 이미지 경로로부터 PIL 이미지를 다시 로드합니다.
    EN: Reloads a PIL image from a spilled image path on disk.
    """
    image = payload.get(image_key)
    if isinstance(image, Image.Image):
        return image
    image_path = str(payload.get(path_key, "")).strip()
    if not image_path or not os.path.exists(image_path):
        return None
    loaded_image = Image.open(image_path)
    loaded_image.load()
    return loaded_image


def _cleanup_deferred_patch_bucket(bucket: dict[str, Any] | None) -> None:
    """KR: 지연 패치 버킷에서 사용된 임시 스필 파일들을 정리합니다.
    EN: Cleans up temporary spill files used in the deferred patch bucket.
    """
    if not isinstance(bucket, dict):
        return
    seen_payloads: set[int] = set()
    seen_paths: set[str] = set()
    for payload in bucket.values():
        if not isinstance(payload, dict):
            continue
        payload_id = id(payload)
        if payload_id in seen_payloads:
            continue
        seen_payloads.add(payload_id)
        for path_key in ("source_atlas_path", "alpha8_linear_source_path"):
            candidate_path = str(payload.get(path_key, "")).strip()
            if candidate_path:
                seen_paths.add(candidate_path)

    for candidate_path in sorted(seen_paths):
        try:
            if os.path.isfile(candidate_path):
                os.remove(candidate_path)
        except Exception:
            pass


def _cleanup_superseded_patch_payload(payload: Any, retained: Any) -> None:
    """Delete spill files unique to an equivalent payload that was deduplicated."""
    if not isinstance(payload, dict) or payload is retained:
        return
    retained_paths = {
        str(retained.get(path_key, "")).strip()
        for path_key in ("source_atlas_path", "alpha8_linear_source_path")
        if isinstance(retained, dict) and str(retained.get(path_key, "")).strip()
    }
    disposable = dict(payload)
    for path_key in ("source_atlas_path", "alpha8_linear_source_path"):
        if str(disposable.get(path_key, "")).strip() in retained_paths:
            disposable.pop(path_key, None)
    _cleanup_deferred_patch_bucket({"payload": disposable})


def _register_deferred_patch(
    patch_map: dict[str, dict[str, Any]] | None,
    target_file_key: str | None,
    object_key: str,
    payload: Any,
    *,
    pending_files: set[str] | None,
    patch_kind: str,
    transaction: _DeferredPatchTransaction | None = None,
) -> bool:
    """KR: 지연 패치(deferred patch)를 패치 맵에 등록합니다. 1패스에서 변경사항을 수집하고 2패스에서 적용하는 구조입니다. 충돌 시 경고를 기록합니다.
    EN: Registers a deferred patch in the patch map. Changes are collected in pass 1 and applied in pass 2. Logs a warning on conflicts.
    """
    normalized_file = _normalize_asset_file_key(target_file_key)
    if not (isinstance(patch_map, dict) and normalized_file and object_key):
        return False
    bucket = patch_map.setdefault(normalized_file, {})
    existing = _lookup_patch_value(bucket, object_key)
    existing_font = (
        str(existing.get("replacement_font", ""))
        if isinstance(existing, dict)
        else ""
    )
    existing_source = (
        str(existing.get("source_entry", ""))
        if isinstance(existing, dict)
        else ""
    )
    new_font = (
        str(payload.get("replacement_font", ""))
        if isinstance(payload, dict)
        else ""
    )
    new_source = (
        str(payload.get("source_entry", ""))
        if isinstance(payload, dict)
        else ""
    )
    if existing is not None and _deferred_patch_fingerprint(
        patch_kind, existing
    ) != _deferred_patch_fingerprint(patch_kind, payload):
        conflict = (
            f"[patch_plan_conflict] kind={patch_kind} file={normalized_file} "
            f"key={object_key} existing={existing_font}@{existing_source} "
            f"new={new_font}@{new_source}"
        )
        _log_warning(conflict)
        if transaction is not None:
            transaction.fail(conflict)
        return False
    retained_payload = existing if existing is not None else payload
    if transaction is not None and not transaction.register_plan(
        patch_kind,
        normalized_file,
        object_key,
        retained_payload,
    ):
        return False
    if existing is not None and existing is not payload:
        _cleanup_superseded_patch_payload(payload, existing)
    _store_patch_value(bucket, object_key, retained_payload)
    if isinstance(pending_files, set) and existing is None:
        pending_files.add(normalized_file)
    return True


def _commit_staged_deferred_patches(
    staged: dict[str, dict[str, Any]],
    target: dict[str, dict[str, Any]] | None,
    *,
    pending_files: set[str] | None,
    patch_kind: str,
    transaction: _DeferredPatchTransaction | None = None,
) -> None:
    if not isinstance(target, dict):
        for bucket in staged.values():
            _cleanup_deferred_patch_bucket(bucket)
        return
    for file_key, bucket in staged.items():
        if not isinstance(bucket, dict):
            continue
        seen_object_keys: set[str] = set()
        for object_key, payload in bucket.items():
            canonical_object_key = str(object_key).lower()
            if canonical_object_key in seen_object_keys:
                continue
            seen_object_keys.add(canonical_object_key)
            if transaction is not None and not transaction.register_plan(
                patch_kind,
                file_key,
                object_key,
                payload,
            ):
                _cleanup_deferred_patch_bucket({object_key: payload})
                continue
            _register_deferred_patch(
                target,
                file_key,
                object_key,
                payload,
                pending_files=pending_files,
                patch_kind=patch_kind,
            )


def _unitypy_supports_streaming_save() -> bool:
    """KR: 현재 UnityPy가 메모리 절감용 save_to() 스트리밍 저장 API를 지원하는지 확인합니다.
    EN: Checks whether the current UnityPy supports the memory-saving save_to() streaming save API.
    """
    return not missing_low_memory_features()


def _ensure_custom_unitypy_streaming_save(lang: Language = "ko") -> None:
    """KR: 스트리밍 저장을 지원하지 않으면 RuntimeError를 발생시킵니다.
    EN: Raises RuntimeError if streaming save is not supported.
    """
    if _unitypy_supports_streaming_save():
        return
    unitypy_path = getattr(UnityPy, "__file__", "")
    unitypy_version = getattr(UnityPy, "__version__", "unknown")
    missing = ", ".join(missing_low_memory_features())
    if lang == "ko":
        raise RuntimeError(
            "현재 UnityPy에는 필요한 저메모리 로드/저장 구현이 없습니다.\n"
            "커스텀 UnityPy 1.25.2 이상을 다시 설치해 주세요.\n"
            f"누락 기능: {missing}\n"
            f"현재 버전/경로: {unitypy_version} / {unitypy_path}"
        )
    raise RuntimeError(
        "The loaded UnityPy does not provide the required low-memory load/save APIs.\n"
        "Reinstall custom UnityPy 1.25.2 or newer.\n"
        f"Missing features: {missing}\n"
        f"Loaded version/path: {unitypy_version} / {unitypy_path}"
    )


def _apply_color_override(current_value: Any, override: JsonDict) -> Any:
    """KR: RGBA 색상 오버라이드를 현재 값에 적용합니다. dict와 객체 속성 모두 처리합니다.
    EN: Applies RGBA color overrides to the current value. Handles both dict and object attributes.
    """
    for attr, key in (("r", "r"), ("g", "g"), ("b", "b"), ("a", "a")):
        if key not in override:
            continue
        try:
            val = float(override[key])
        except Exception:
            continue
        if isinstance(current_value, dict):
            current_value[key] = val
        if hasattr(current_value, attr):
            try:
                setattr(current_value, attr, val)
            except Exception:
                pass
    return current_value


def _texture_ref_to_dict(texture_ref: Any) -> JsonDict:
    """KR: 텍스처 참조를 m_FileID/m_PathID 딕셔너리로 변환합니다.
    EN: Converts a texture reference to an m_FileID/m_PathID dictionary.
    """
    if isinstance(texture_ref, dict):
        file_id = int(texture_ref.get("m_FileID", 0) or 0)
        path_id = int(texture_ref.get("m_PathID", 0) or 0)
        return {"m_FileID": file_id, "m_PathID": path_id}
    file_id = int(getattr(texture_ref, "m_FileID", 0) or 0)
    path_id = int(getattr(texture_ref, "m_PathID", 0) or 0)
    return {"m_FileID": file_id, "m_PathID": path_id}


def _extract_texture_ref_from_tex_env(env_value: Any) -> JsonDict:
    """KR: TexEnv 항목에서 m_Texture 참조를 딕셔너리로 추출합니다.
    EN: Extracts the m_Texture reference as a dictionary from a TexEnv entry.
    """
    if isinstance(env_value, dict):
        return _texture_ref_to_dict(env_value.get("m_Texture"))
    tex = getattr(env_value, "m_Texture", None)
    return _texture_ref_to_dict(tex)


def _color_value_to_dict(value: Any, default: JsonDict) -> JsonDict:
    """KR: 색상 값을 RGBA 딕셔너리로 변환합니다. 누락된 채널은 기본값으로 채웁니다.
    EN: Converts a color value to an RGBA dictionary. Missing channels are filled with defaults.
    """
    if isinstance(value, dict):
        return {
            "r": float(value.get("r", default["r"])),
            "g": float(value.get("g", default["g"])),
            "b": float(value.get("b", default["b"])),
            "a": float(value.get("a", default["a"])),
        }
    out = dict(default)
    for key in ("r", "g", "b", "a"):
        attr = getattr(value, key, None)
        if attr is not None:
            try:
                out[key] = float(attr)
            except Exception:
                pass
    return out


def _build_tex_env_entry(texture_ref: JsonDict) -> JsonDict:
    """KR: 텍스처 참조로부터 TexEnv 항목을 구성합니다. Scale=(1,1), Offset=(0,0) 기본값을 사용합니다.
    EN: Builds a TexEnv entry from a texture reference. Uses defaults Scale=(1,1), Offset=(0,0).
    """
    return {
        "m_Texture": {
            "m_FileID": int(texture_ref.get("m_FileID", 0) or 0),
            "m_PathID": int(texture_ref.get("m_PathID", 0) or 0),
        },
        "m_Scale": {"x": 1.0, "y": 1.0},
        "m_Offset": {"x": 0.0, "y": 0.0},
    }


def _prune_material_saved_properties_for_raster(
    parse_dict: Any,
    color_overrides: dict[str, JsonDict],
) -> bool:
    """KR: 래스터 폰트용으로 머티리얼의 SavedProperties를 최소 속성 세트로 정리합니다.
    EN: Prunes material SavedProperties to a minimal property set for raster fonts.
    """
    saved_props = getattr(parse_dict, "m_SavedProperties", None)
    if saved_props is None:
        return False

    tex_envs = getattr(saved_props, "m_TexEnvs", None)
    main_tex_ref: JsonDict = {"m_FileID": 0, "m_PathID": 0}
    face_tex_ref: JsonDict = {"m_FileID": 0, "m_PathID": 0}
    if isinstance(tex_envs, list):
        for entry in tex_envs:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            prop_name = str(entry[0])
            env_value = entry[1]
            if prop_name == "_MainTex":
                main_tex_ref = _extract_texture_ref_from_tex_env(env_value)
            elif prop_name == "_FaceTex":
                face_tex_ref = _extract_texture_ref_from_tex_env(env_value)

    new_tex_envs: list[tuple[str, JsonDict]] = [
        ("_FaceTex", _build_tex_env_entry(face_tex_ref)),
        ("_MainTex", _build_tex_env_entry(main_tex_ref)),
    ]
    new_floats: list[tuple[str, float]] = [
        ("_ColorMask", 15.0),
        ("_CullMode", 0.0),
        ("_MaskSoftnessX", 0.0),
        ("_MaskSoftnessY", 0.0),
        ("_Stencil", 0.0),
        ("_StencilComp", 8.0),
        ("_StencilOp", 0.0),
        ("_StencilReadMask", 255.0),
        ("_StencilWriteMask", 255.0),
        ("_VertexOffsetX", 0.0),
        ("_VertexOffsetY", 0.0),
    ]

    color_map: dict[str, Any] = {}
    old_colors = getattr(saved_props, "m_Colors", None)
    if isinstance(old_colors, list):
        for entry in old_colors:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            color_map[str(entry[0])] = entry[1]

    clip_rect = _color_value_to_dict(
        color_map.get("_ClipRect"),
        {"r": -32767.0, "g": -32767.0, "b": 32767.0, "a": 32767.0},
    )
    face_color_value = _color_value_to_dict(
        color_map.get("_FaceColor"),
        {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
    )
    face_override = color_overrides.get("_FaceColor")
    if isinstance(face_override, dict):
        face_color_value = _apply_color_override(face_color_value, face_override)

    new_colors: list[tuple[str, JsonDict]] = [
        ("_ClipRect", clip_rect),
        ("_FaceColor", face_color_value),
    ]

    saved_props.m_TexEnvs = new_tex_envs
    if hasattr(saved_props, "m_Ints"):
        try:
            saved_props.m_Ints = []
        except Exception:
            pass
    saved_props.m_Floats = new_floats
    saved_props.m_Colors = new_colors
    return True


def _apply_material_replacement_to_object(parse_dict: Any, mat_info: JsonDict) -> bool:
    """KR: 머티리얼 교체 정보를 파싱된 객체에 적용합니다. float/color 오버라이드, 외곽선 비율, 스타일 보존 등을 처리합니다.
    EN: Applies material replacement info to the parsed object. Handles float/color overrides, outline ratio, style preservation, etc.
    """
    changed = False
    float_overrides_raw = mat_info.get("float_overrides", {})
    float_overrides = (
        float_overrides_raw if isinstance(float_overrides_raw, dict) else {}
    )
    color_overrides_raw = mat_info.get("color_overrides", {})
    color_overrides = (
        color_overrides_raw if isinstance(color_overrides_raw, dict) else {}
    )
    try:
        outline_ratio = float(mat_info.get("outline_ratio", 1.0))
    except Exception:
        outline_ratio = 1.0
    if outline_ratio <= 0:
        outline_ratio = 1.0
    outline_fallback_used = False
    preserve_game_style = bool(mat_info.get("preserve_game_style", False))
    try:
        style_padding_scale_ratio = float(mat_info.get("style_padding_scale_ratio", 1.0))
    except Exception:
        style_padding_scale_ratio = 1.0
    if style_padding_scale_ratio <= 0:
        style_padding_scale_ratio = 1.0
    prune_raster_material = bool(mat_info.get("prune_raster_material", False))
    replacement_padding = float(mat_info.get("replacement_padding", 0) or 0)
    gradient_scale = mat_info.get("gs")
    texture_h_raw = mat_info.get("h")
    texture_w_raw = mat_info.get("w")
    try:
        texture_h = float(texture_h_raw) if texture_h_raw is not None else None
    except Exception:
        texture_h = None
    try:
        texture_w = float(texture_w_raw) if texture_w_raw is not None else None
    except Exception:
        texture_w = None

    saved_props = getattr(parse_dict, "m_SavedProperties", None)
    if saved_props is None:
        return False

    if prune_raster_material:
        if _prune_material_saved_properties_for_raster(parse_dict, color_overrides):
            changed = True
    else:
        float_props = getattr(saved_props, "m_Floats", None)
        if isinstance(float_props, list):
            existing_float_map: dict[str, float] = {}
            for entry in float_props:
                if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                    continue
                try:
                    existing_float_map[str(entry[0])] = float(entry[1])
                except Exception:
                    continue

            has_texture_height = False
            has_texture_width = False
            has_gradient_scale = False
            for i in range(len(float_props)):
                entry = float_props[i]
                if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                    continue
                prop_name = str(entry[0])
                if prop_name == "_GradientScale":
                    candidate: float | None = None
                    if prop_name in float_overrides:
                        try:
                            candidate = float(float_overrides[prop_name])
                        except Exception:
                            candidate = None
                    elif gradient_scale is not None:
                        try:
                            candidate = float(gradient_scale)
                        except Exception:
                            candidate = None
                    if candidate is not None:
                        # KR: _GradientScale은 교체 아틀라스의 padding 기반 값을 강제 적용합니다.
                        # preserve_gradient_floor 로직은 교체 아틀라스와 불일치를 유발하므로 제거되었습니다.
                        # EN: _GradientScale is force-set to the padding-based value of the replacement atlas.
                        # The preserve_gradient_floor logic was removed as it caused mismatch with the replacement atlas.
                        float_props[i] = ("_GradientScale", candidate)
                        has_gradient_scale = True
                        changed = True
                elif preserve_game_style and prop_name in _MATERIAL_STYLE_FLOAT_KEYS:
                    candidate = existing_float_map.get(prop_name)
                    if candidate is None:
                        continue
                    if prop_name in _MATERIAL_STYLE_PADDING_SCALE_KEYS:
                        candidate = float(candidate * style_padding_scale_ratio)
                    if prop_name in _MATERIAL_OUTLINE_RATIO_KEYS:
                        candidate = float(candidate * outline_ratio)
                    float_props[i] = (prop_name, float(candidate))
                    changed = True
                elif prop_name in _MATERIAL_OUTLINE_RATIO_KEYS:
                    candidate: float | None = None
                    existing_value: float | None = None
                    try:
                        existing_value = float(entry[1])
                    except Exception:
                        existing_value = None
                    if prop_name in float_overrides:
                        try:
                            candidate = float(float_overrides[prop_name])
                        except Exception:
                            candidate = None
                        if (
                            outline_ratio != 1.0
                            and candidate is not None
                            and abs(candidate) <= 1e-9
                        ):
                            if existing_value is not None and abs(existing_value) > 1e-9:
                                candidate = existing_value
                                outline_fallback_used = True
                            elif prop_name == "_OutlineWidth":
                                baseline_gradient_scale = None
                                try:
                                    if "_GradientScale" in float_overrides:
                                        baseline_gradient_scale = float(
                                            float_overrides["_GradientScale"]
                                        )
                                    elif gradient_scale is not None:
                                        baseline_gradient_scale = float(gradient_scale)
                                    else:
                                        baseline_gradient_scale = existing_float_map.get(
                                            "_GradientScale"
                                        )
                                except Exception:
                                    baseline_gradient_scale = None
                                if (
                                    baseline_gradient_scale is not None
                                    and baseline_gradient_scale > 0
                                ):
                                    candidate = 1.0 / baseline_gradient_scale
                                    outline_fallback_used = True
                    elif outline_ratio != 1.0:
                        candidate = existing_value
                        if (
                            prop_name == "_OutlineWidth"
                            and candidate is not None
                            and abs(candidate) <= 1e-9
                        ):
                            baseline_gradient_scale = None
                            try:
                                if "_GradientScale" in float_overrides:
                                    baseline_gradient_scale = float(
                                        float_overrides["_GradientScale"]
                                    )
                                elif gradient_scale is not None:
                                    baseline_gradient_scale = float(gradient_scale)
                                else:
                                    baseline_gradient_scale = existing_float_map.get(
                                        "_GradientScale"
                                    )
                            except Exception:
                                baseline_gradient_scale = None
                            if (
                                baseline_gradient_scale is not None
                                and baseline_gradient_scale > 0
                            ):
                                candidate = 1.0 / baseline_gradient_scale
                                outline_fallback_used = True
                    if candidate is not None:
                        float_props[i] = (prop_name, float(candidate * outline_ratio))
                        changed = True
                elif prop_name == "_TextureHeight" and texture_h is not None:
                    # KR: _TextureHeight는 실제 아틀라스 크기가 float_overrides보다 우선합니다.
                    # EN: For _TextureHeight, the actual atlas size takes priority over float_overrides.
                    float_props[i] = ("_TextureHeight", texture_h)
                    has_texture_height = True
                    changed = True
                elif prop_name == "_TextureWidth" and texture_w is not None:
                    float_props[i] = ("_TextureWidth", texture_w)
                    has_texture_width = True
                    changed = True
                elif prop_name in float_overrides:
                    float_props[i] = (prop_name, float(float_overrides[prop_name]))
                    changed = True
                if prop_name == "_TextureHeight":
                    has_texture_height = True
                elif prop_name == "_TextureWidth":
                    has_texture_width = True
                elif prop_name == "_GradientScale":
                    has_gradient_scale = True
            if texture_h is not None and not has_texture_height:
                float_props.append(("_TextureHeight", texture_h))
                changed = True
            if texture_w is not None and not has_texture_width:
                float_props.append(("_TextureWidth", texture_w))
                changed = True
            if gradient_scale is not None and not has_gradient_scale:
                float_props.append(("_GradientScale", float(gradient_scale)))
                changed = True

            # KR: _ScaleRatioA를 교체 아틀라스의 padding/GradientScale로 재계산합니다.
            # TMP에서 ScaleRatioA = padding / GradientScale이며, 이 값이 불일치하면 외곽선/그림자 크기가 틀어집니다.
            # EN: Recalculates _ScaleRatioA using the replacement atlas padding/GradientScale.
            # In TMP, ScaleRatioA = padding / GradientScale; mismatch causes incorrect outline/shadow sizes.
            if replacement_padding > 0:
                final_gs = None
                for _fp in float_props:
                    if isinstance(_fp, (list, tuple)) and len(_fp) >= 2 and _fp[0] == "_GradientScale":
                        try:
                            final_gs = float(_fp[1])
                        except Exception:
                            pass
                        break
                if final_gs and final_gs > 0:
                    new_scale_ratio_a = replacement_padding / final_gs
                    for k, fp in enumerate(float_props):
                        if isinstance(fp, (list, tuple)) and len(fp) >= 2 and fp[0] == "_ScaleRatioA":
                            float_props[k] = ("_ScaleRatioA", float(new_scale_ratio_a))
                            changed = True
                            break

            if outline_fallback_used:
                logger.debug(
                    "outline_ratio used original material baseline because replacement outline values were zero: %s",
                    mat_info.get("source_entry", ""),
                )

        color_props = getattr(saved_props, "m_Colors", None)
        if isinstance(color_props, list) and color_overrides:
            for i in range(len(color_props)):
                color_name = color_props[i][0]
                if preserve_game_style and str(color_name) in _MATERIAL_STYLE_COLOR_KEYS:
                    continue
                override = color_overrides.get(color_name)
                if not isinstance(override, dict):
                    continue
                current_value = color_props[i][1]
                color_props[i] = (
                    color_name,
                    _apply_color_override(current_value, override),
                )
                changed = True

    if bool(mat_info.get("reset_keywords", False)):
        if hasattr(parse_dict, "m_ShaderKeywords"):
            try:
                parse_dict.m_ShaderKeywords = ""
                changed = True
            except Exception:
                pass
        if hasattr(parse_dict, "m_ValidKeywords"):
            try:
                parse_dict.m_ValidKeywords = []
                changed = True
            except Exception:
                pass
        if hasattr(parse_dict, "m_InvalidKeywords"):
            try:
                parse_dict.m_InvalidKeywords = []
                changed = True
            except Exception:
                pass
    return changed


def find_assets_files(
    game_path: str,
    lang: Language = "ko",
    target_files: set[str] | None = None,
    exclude_exts: set[str] | None = None,
) -> list[str]:
    """Collect candidate Unity assets through the scanner module."""
    return _find_assets_files_impl(
        game_path,
        lang=lang,
        target_files=target_files,
        exclude_exts=exclude_exts,
        data_path_resolver=get_data_path,
        log_console=_log_console,
    )


def _scan_fonts_from_env(
    env: Any,
    file_name: str,
    lang: Language = "ko",
    detect_ps5_swizzle: bool = False,
    scan_ttf: bool = True,
    scan_sdf: bool = True,
    phase_callback: Callable[[str, JsonDict], None] | None = None,
) -> dict[str, list[JsonDict]]:
    """Compatibility facade for in-process environment scanning."""
    return _scan_fonts_from_env_impl(
        env,
        file_name,
        lang=lang,
        detect_ps5_swizzle=detect_ps5_swizzle,
        scan_ttf=scan_ttf,
        scan_sdf=scan_sdf,
        phase_callback=phase_callback,
        log_console=_log_console,
        debug_log=debug_parse_log,
    )


def _scan_fonts_in_asset_file(
    assets_file: str,
    generator: TypeTreeGenerator | None,
    lang: Language = "ko",
    detect_ps5_swizzle: bool = False,
    scan_ttf: bool = True,
    scan_sdf: bool = True,
    phase_callback: Callable[[str, JsonDict], None] | None = None,
) -> tuple[dict[str, list[JsonDict]], str | None]:
    """Compatibility facade preserving runtime dependency patch points."""
    return _scan_fonts_in_asset_file_impl(
        assets_file,
        generator,
        lang=lang,
        detect_ps5_swizzle=detect_ps5_swizzle,
        scan_ttf=scan_ttf,
        scan_sdf=scan_sdf,
        phase_callback=phase_callback,
        load_environment=load_unitypy,
        close_environment=close_unitypy_env,
        scan_environment=_scan_fonts_from_env,
    )


def get_compile_method(datapath: str) -> str:
    """KR: 데이터 폴더의 컴파일 방식을 Mono/Il2cpp로 판별합니다.
    EN: Determines the compile method (Mono/Il2cpp) of the data folder.
    """
    if "Managed" in os.listdir(datapath):
        return "Mono"
    else:
        return "Il2cpp"


def _create_generator(
    unity_version: str,
    game_path: str,
    data_path: str,
    compile_method: str,
    lang: Language = "ko",
) -> TypeTreeGenerator:
    """KR: 타입트리 생성기를 구성하고 Mono/Il2cpp 메타데이터를 로드합니다.
    EN: Configures the TypeTree generator and loads Mono/Il2cpp metadata.
    """
    generator = TypeTreeGenerator(unity_version)
    if compile_method == "Mono":
        managed_dir = os.path.join(data_path, "Managed")
        for fn in os.listdir(managed_dir):
            if not fn.endswith(".dll"):
                continue
            try:
                with open(os.path.join(managed_dir, fn), "rb") as f:
                    generator.load_dll(f.read())
            except Exception as e:
                if lang == "ko":
                    _log_console(f"[generator] DLL 로드 실패: {fn} ({e})")
                else:
                    _log_console(f"[generator] Failed to load DLL: {fn} ({e})")
    else:
        il2cpp_path = os.path.join(game_path, "GameAssembly.dll")
        with open(il2cpp_path, "rb") as f:
            il2cpp = f.read()
        metadata_path = os.path.join(
            data_path, "il2cpp_data", "Metadata", "global-metadata.dat"
        )
        with open(metadata_path, "rb") as f:
            metadata = f.read()
        generator.load_il2cpp(il2cpp, metadata)
    return generator


def _build_scan_worker_server_command(
    game_path: str,
    *,
    lang: Language,
    detect_ps5_swizzle: bool,
    scan_ttf: bool,
    scan_sdf: bool,
) -> list[str]:
    """KR: 현재 소스/동결 실행 환경에 맞는 영구 워커 명령을 구성합니다.
    EN: Build the persistent-worker command for source or frozen execution.
    """
    if getattr(sys, "frozen", False):
        command = [sys.executable]
    else:
        command = [sys.executable, os.path.abspath(__file__)]
    command.extend(
        [
            "--gamepath",
            game_path,
            "--_scan-worker-server",
            "--_scan-worker-lang",
            lang,
        ]
    )
    if detect_ps5_swizzle:
        command.append("--ps5-swizzle")
    if scan_ttf and not scan_sdf:
        command.append("--_scan-ttf-only")
    elif scan_sdf and not scan_ttf:
        command.append("--_scan-sdf-only")
    return command


def _normalize_scan_pool_result(
    result: ScanPoolResult,
) -> tuple[dict[str, list[JsonDict]], str | None]:
    """KR: 프로토콜 payload를 기존 스캔 반환 형식으로 정규화합니다.
    EN: Normalize a protocol payload to the legacy scan return shape.
    """
    payload = result.payload if isinstance(result.payload, dict) else {}
    ttf_raw = payload.get("ttf", [])
    sdf_raw = payload.get("sdf", [])
    scanned: dict[str, list[JsonDict]] = {
        "ttf": list(ttf_raw) if isinstance(ttf_raw, list) else [],
        "sdf": list(sdf_raw) if isinstance(sdf_raw, list) else [],
    }
    payload_error = payload.get("error")
    errors = [
        text
        for text in (
            result.error,
            result.warning,
            payload_error if isinstance(payload_error, str) else None,
        )
        if isinstance(text, str) and text.strip()
    ]
    return scanned, " | ".join(errors) if errors else None


def scan_fonts(
    game_path: str,
    lang: Language = "ko",
    target_files: set[str] | None = None,
    exclude_exts: set[str] | None = None,
    isolate_files: bool = True,
    scan_jobs: int = 1,
    ps5_swizzle: bool = False,
    scan_ttf: bool = True,
    scan_sdf: bool = True,
    scan_stall_seconds: float = DEFAULT_STALL_SECONDS,
) -> dict[str, list[JsonDict]]:
    """KR: 게임 에셋을 스캔해 TTF/SDF 폰트 목록을 반환합니다.

    target_files가 있으면 해당 파일만 스캔합니다.
    exclude_exts가 있으면 해당 확장자는 스캔에서 제외합니다.
    isolate_files=True면 재사용 워커 프로세스로 스캔해 크래시를 격리합니다.
    scan_jobs는 영구 워커 풀의 크기입니다.
    scan_stall_seconds는 CPU/I/O/단계 진행이 모두 멈춘 시간만 측정하며,
    전체 파일 처리 시간에는 제한을 두지 않습니다.
    EN: Scans game assets and returns a list of TTF/SDF fonts.

    If target_files is provided, only those files are scanned.
    If exclude_exts is provided, those extensions are excluded from scanning.
    If isolate_files=True, scans via reusable worker processes to isolate crashes.
    scan_jobs controls the persistent worker pool size.
    scan_stall_seconds measures only CPU/I/O/protocol inactivity; it does not
    impose a total per-file wall-clock limit.
    """
    scan_ttf = bool(scan_ttf)
    scan_sdf = bool(scan_sdf)
    if not scan_ttf and not scan_sdf:
        return {"ttf": [], "sdf": []}

    data_path = get_data_path(game_path, lang=lang)
    assets_files = find_assets_files(
        game_path,
        lang=lang,
        target_files=target_files,
        exclude_exts=exclude_exts,
    )
    generator: TypeTreeGenerator | None = None
    # KR: 격리 워커는 자체 generator를 만들며, TTF-only 스캔은 TypeTree가 필요 없습니다.
    # EN: Isolated workers create their own generator, and TTF-only scans need no TypeTree.
    if not isolate_files and scan_sdf:
        unity_version = get_unity_version(game_path, lang=lang)
        compile_method = get_compile_method(data_path)
        generator = _create_generator(
            unity_version, game_path, data_path, compile_method, lang=lang
        )

    fonts: dict[str, list[JsonDict]] = {
        "ttf": [],
        "sdf": [],
    }

    total_files = len(assets_files)
    try:
        scan_jobs = int(scan_jobs)
    except Exception:
        scan_jobs = 1
    if scan_jobs < 1:
        scan_jobs = 1
    try:
        scan_stall_seconds = float(scan_stall_seconds)
    except Exception:
        scan_stall_seconds = DEFAULT_STALL_SECONDS
    if scan_stall_seconds < 0:
        scan_stall_seconds = 0.0
    if lang == "ko":
        if target_files:
            _log_console(
                f"[scan_fonts] --target-file 기준 스캔 시작: {total_files}개 파일"
            )
        else:
            _log_console(f"[scan_fonts] 전체 스캔 시작: {total_files}개 파일")
    else:
        if target_files:
            _log_console(
                f"[scan_fonts] Starting target-file scan: {total_files} file(s)"
            )
        else:
            _log_console(f"[scan_fonts] Starting full scan: {total_files} file(s)")

    if isolate_files and total_files > 0:
        max_workers = min(scan_jobs, total_files)
        if lang == "ko":
            _log_console(
                f"[scan_fonts] 영구 워커 모드: {max_workers}개 "
                f"(워커당 최대 {DEFAULT_MAX_JOBS_PER_WORKER}개 파일 재사용)"
            )
        else:
            _log_console(
                f"[scan_fonts] Persistent worker mode: {max_workers} "
                f"(recycle after {DEFAULT_MAX_JOBS_PER_WORKER} files per worker)"
            )
        if scan_stall_seconds > 0:
            if lang == "ko":
                _log_console(
                    "[scan_fonts] 무활동 정지 판정: "
                    f"{float(scan_stall_seconds):g}초 "
                    "(총 처리시간 제한 아님)"
                )
            else:
                _log_console(
                    "[scan_fonts] Inactivity stall threshold: "
                    f"{float(scan_stall_seconds):g}s "
                    "(not a total runtime limit)"
                )

        command = _build_scan_worker_server_command(
            game_path,
            lang=lang,
            detect_ps5_swizzle=ps5_swizzle,
            scan_ttf=scan_ttf,
            scan_sdf=scan_sdf,
        )

        def _report_progress(
            completed: int,
            total: int,
            result: ScanPoolResult,
        ) -> None:
            file_name = os.path.basename(result.asset_path)
            if lang == "ko":
                _log_console(f"[scan_fonts] 진행 {completed}/{total}: {file_name}")
            else:
                _log_console(
                    f"[scan_fonts] Progress {completed}/{total}: {file_name}"
                )

        pool_results = PersistentScanWorkerPool(
            command,
            worker_count=max_workers,
            max_retries=1,
            stall_seconds=scan_stall_seconds,
            max_jobs_per_worker=DEFAULT_MAX_JOBS_PER_WORKER,
            progress_callback=_report_progress,
        ).scan(assets_files)

        for result in pool_results:
            processed_file_name = os.path.basename(result.asset_path)
            scanned, worker_error = _normalize_scan_pool_result(result)
            if worker_error:
                if lang == "ko":
                    _log_console(
                        f"[scan_fonts] 워커 경고: {processed_file_name} | {worker_error}"
                    )
                else:
                    _log_console(
                        f"[scan_fonts] Worker warning: {processed_file_name} | {worker_error}"
                    )
            _log_scan_result_details(
                processed_file_name or f"index_{result.index}",
                scanned,
            )
            fonts["ttf"].extend(scanned.get("ttf", []))
            fonts["sdf"].extend(scanned.get("sdf", []))
    else:
        for idx, assets_file in enumerate(assets_files, start=1):
            fn = os.path.basename(assets_file)
            if lang == "ko":
                _log_console(f"[scan_fonts] 진행 {idx}/{total_files}: {fn}")
            else:
                _log_console(f"[scan_fonts] Progress {idx}/{total_files}: {fn}")

            scanned, load_error = _scan_fonts_in_asset_file(
                assets_file,
                generator,
                lang=lang,
                detect_ps5_swizzle=ps5_swizzle,
                scan_ttf=scan_ttf,
                scan_sdf=scan_sdf,
            )
            if load_error:
                _log_console(f"[scan_fonts] {load_error}")
                continue
            _log_scan_result_details(fn, scanned)
            fonts["ttf"].extend(scanned.get("ttf", []))
            fonts["sdf"].extend(scanned.get("sdf", []))

    return fonts


def parse_fonts(
    game_path: str,
    lang: Language = "ko",
    target_files: set[str] | None = None,
    exclude_exts: set[str] | None = None,
    scan_jobs: int = 1,
    ps5_swizzle: bool = False,
    scan_stall_seconds: float = DEFAULT_STALL_SECONDS,
) -> str:
    """KR: 스캔한 폰트를 JSON으로 저장하고 결과 파일 경로를 반환합니다.

    target_files가 있으면 해당 파일만 파싱합니다.
    exclude_exts가 있으면 해당 확장자는 스캔에서 제외합니다.
    EN: Saves scanned fonts as JSON and returns the result file path.

    If target_files is provided, only those files are parsed.
    If exclude_exts is provided, those extensions are excluded from scanning.
    """
    # KR: parse 모드는 재사용 워커 풀로 스캔해 UnityPy 하드 크래시를 격리합니다.
    # EN: Parse mode scans via a reusable worker pool to isolate UnityPy hard crashes.
    fonts = scan_fonts(
        game_path,
        lang=lang,
        target_files=target_files,
        exclude_exts=exclude_exts,
        isolate_files=True,
        scan_jobs=scan_jobs,
        ps5_swizzle=ps5_swizzle,
        scan_stall_seconds=scan_stall_seconds,
    )
    game_name = os.path.basename(game_path)
    output_file = os.path.join(get_script_dir(), f"{game_name}.json")

    result: dict[str, JsonDict] = {}

    for font in fonts["ttf"]:
        key = (
            f"{font['file']}|{font['assets_name']}|{font['name']}|TTF|{font['path_id']}"
        )
        result[key] = {
            "File": font["file"],
            "assets_name": font["assets_name"],
            "Path_ID": font["path_id"],
            "Type": "TTF",
            "Name": font["name"],
            "Replace_to": "",
        }

    for font in fonts["sdf"]:
        key = (
            f"{font['file']}|{font['assets_name']}|{font['name']}|SDF|{font['path_id']}"
        )
        if ps5_swizzle:
            swizzle_flag = "True" if parse_bool_flag(font.get("swizzle")) else "False"
            process_swizzle_flag = (
                "True" if parse_bool_flag(font.get("process_swizzle")) else "False"
            )
            entry: JsonDict = {
                "File": font["file"],
                "assets_name": font["assets_name"],
                "Path_ID": font["path_id"],
                "Type": "SDF",
                "Name": font["name"],
                "force_raster": "False",
                "swizzle": swizzle_flag,
                "process_swizzle": process_swizzle_flag,
                "Replace_to": "",
            }
        else:
            entry = {
                "File": font["file"],
                "assets_name": font["assets_name"],
                "Path_ID": font["path_id"],
                "Type": "SDF",
                "Name": font["name"],
                "force_raster": "False",
                "Replace_to": "",
            }
        result[key] = entry

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    if lang == "ko":
        _log_console(f"폰트 정보가 '{output_file}'에 저장되었습니다.")
        _log_console(f"  - TTF 폰트: {len(fonts['ttf'])}개")
        _log_console(f"  - SDF 폰트: {len(fonts['sdf'])}개")
    else:
        _log_console(f"Font information saved to '{output_file}'.")
        _log_console(f"  - TTF fonts: {len(fonts['ttf'])}")
        _log_console(f"  - SDF fonts: {len(fonts['sdf'])}")
    return output_file


def _format_byte_size(num_bytes: int) -> str:
    size = float(max(0, int(num_bytes or 0)))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{int(num_bytes or 0)} B"


def _dedupe_preserve_order_str(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in values:
        key = str(item).strip()
        if not key:
            continue
        lowered = key.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(key)
    return ordered


def _build_font_asset_name_candidates(
    normalized: str,
    prefer_raster: bool = False,
) -> tuple[list[str], list[str]]:
    raw_name = str(normalized).strip()

    def _strip_render_suffix(name: str) -> str:
        if name.endswith(" SDF"):
            return name[: -len(" SDF")]
        if name.endswith(" Raster"):
            return name[: -len(" Raster")]
        return name

    base_name = _strip_render_suffix(raw_name)
    if prefer_raster:
        name_candidates = _dedupe_preserve_order_str(
            [raw_name, f"{base_name} Raster", f"{base_name} SDF"]
        )
    else:
        name_candidates = _dedupe_preserve_order_str(
            [raw_name, f"{base_name} SDF", f"{base_name} Raster"]
        )

    font_name_candidates = _dedupe_preserve_order_str(
        [raw_name, base_name] + name_candidates
    )
    return font_name_candidates, name_candidates


_BULK_SDF_PADDING_VARIANTS = (5, 7, 15)


def _select_builtin_bulk_padding_variant(
    normalized: str,
    source_padding: float | int | None,
) -> int | None:
    base_name = normalize_font_name(normalized).strip().lower()
    if base_name not in {"nanumgothic", "mulmaru"}:
        return None
    try:
        numeric_padding = float(source_padding) if source_padding is not None else 0.0
    except Exception:
        numeric_padding = 0.0
    if numeric_padding <= 0:
        return None
    return min(
        _BULK_SDF_PADDING_VARIANTS,
        key=lambda value: (abs(float(value) - numeric_padding), -int(value)),
    )


def select_replacement_asset_padding(
    replacement_font: str,
    source_padding_hint: float | int | None,
    selected_builtin_padding: int | None,
) -> int | None:
    if selected_builtin_padding is not None:
        return int(selected_builtin_padding)
    if _resolve_ttf_source_path(str(replacement_font)) is None:
        return None
    try:
        numeric_padding = int(round(float(source_padding_hint or 0)))
    except Exception:
        numeric_padding = 0
    return numeric_padding if numeric_padding > 0 else 7


def _iter_kr_asset_roots(
    kr_assets: str,
    padding_variant: int | None = None,
) -> list[str]:
    roots: list[str] = []
    if padding_variant is not None:
        roots.append(os.path.join(kr_assets, f"Padding_{int(padding_variant)}"))
    roots.append(kr_assets)
    return roots


def _candidate_font_file_paths(source: str, script_dir: str) -> list[str]:
    raw = strip_wrapping_quotes_repeated(str(source))
    normalized = normalize_font_name(raw)
    source_names = _dedupe_preserve_order_str([raw, normalized])
    roots = _dedupe_preserve_order_str(
        [
            "",
            os.getcwd(),
            script_dir,
            os.path.join(script_dir, "KR_ASSETS"),
        ]
    )
    candidates: list[str] = []

    for name in source_names:
        if not name:
            continue
        direct_names = [name]
        base, ext = os.path.splitext(name)
        if ext.lower() not in {".ttf", ".otf"}:
            direct_names.extend([f"{name}.ttf", f"{name}.otf"])
        for direct_name in direct_names:
            if os.path.isabs(direct_name):
                candidates.append(direct_name)
                continue
            for root in roots:
                if root:
                    candidates.append(os.path.join(root, direct_name))
                else:
                    candidates.append(direct_name)

    return _dedupe_preserve_order_str(candidates)


def _resolve_ttf_source_path(source: str, script_dir: str | None = None) -> str | None:
    if script_dir is None:
        script_dir = get_script_dir()
    for candidate in _candidate_font_file_paths(source, script_dir):
        if not os.path.exists(candidate):
            continue
        if os.path.splitext(candidate)[1].lower() not in {".ttf", ".otf"}:
            continue
        return os.path.abspath(candidate)
    return None


def _fallback_font_display_name(fallback_name: str) -> str:
    raw = strip_wrapping_quotes_repeated(str(fallback_name)).strip()
    if not raw:
        return "Font"
    base = os.path.basename(raw)
    if base:
        raw = base
    return normalize_font_name(raw).strip() or "Font"


def _dedupe_nonempty_strings(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _read_ttf_name_records(name_table: Any, name_ids: set[int]) -> list[str]:
    names: list[str] = []
    for record in getattr(name_table, "names", []) or []:
        if int(getattr(record, "nameID", -1)) not in name_ids:
            continue
        try:
            value = record.toUnicode()
        except Exception:
            continue
        names.append(value)
    return _dedupe_nonempty_strings(names)


def read_ttf_font_metadata(
    ttf_data: bytes,
    fallback_name: str = "",
    font_size: float | int = 16.0,
) -> JsonDict:
    fallback = _fallback_font_display_name(fallback_name)
    metadata: JsonDict = {
        "parsed": False,
        "font_names": [],
        "ascent": None,
        "descent": None,
        "line_spacing": None,
    }
    if TTFont is None:
        return metadata

    try:
        size = float(font_size or 16.0)
    except Exception:
        size = 16.0
    if size <= 0:
        size = 16.0

    try:
        with TTFont(io.BytesIO(bytes(ttf_data)), lazy=True) as ttf:
            name_table = ttf.get("name")
            name_candidates: list[str] = []
            if name_table is not None:
                name_candidates.extend(_read_ttf_name_records(name_table, {16}))
                try:
                    name_candidates.append(name_table.getBestFamilyName())
                except Exception:
                    pass
                name_candidates.extend(_read_ttf_name_records(name_table, {1}))
            metadata["font_names"] = _dedupe_nonempty_strings(
                name_candidates + [fallback]
            )

            units_per_em = 1000
            head = ttf.get("head")
            if head is not None:
                units_per_em = int(getattr(head, "unitsPerEm", units_per_em) or 1000)
            units_per_em = max(1, units_per_em)

            ascent = descent = line_gap = None
            hhea = ttf.get("hhea")
            if hhea is not None:
                ascent = float(getattr(hhea, "ascent", 0) or 0)
                descent = float(getattr(hhea, "descent", 0) or 0)
                line_gap = float(getattr(hhea, "lineGap", 0) or 0)
            if ascent is None or descent is None:
                os2_table = ttf.get("OS/2")
                if os2_table is not None:
                    ascent = float(getattr(os2_table, "usWinAscent", 0) or 0)
                    descent = -float(getattr(os2_table, "usWinDescent", 0) or 0)
                    line_gap = float(getattr(os2_table, "sTypoLineGap", 0) or 0)

            if ascent is not None and descent is not None:
                if line_gap is None:
                    line_gap = 0.0
                scale = size / float(units_per_em)
                metadata["ascent"] = float(ascent * scale)
                metadata["descent"] = float(descent * scale)
                metadata["line_spacing"] = float((ascent - descent + line_gap) * scale)
            metadata["parsed"] = True
    except Exception:
        return metadata

    return metadata


def apply_ttf_metadata_to_font(
    font: Any,
    ttf_data: bytes,
    fallback_name: str = "",
) -> JsonDict:
    metadata = read_ttf_font_metadata(
        ttf_data,
        fallback_name=fallback_name,
        font_size=getattr(font, "m_FontSize", 16.0),
    )
    font_names = metadata.get("font_names")
    if (
        metadata.get("parsed")
        and isinstance(font_names, list)
        and font_names
        and hasattr(font, "m_FontNames")
    ):
        font.m_FontNames = font_names
    for attr, key in (
        ("m_Ascent", "ascent"),
        ("m_Descent", "descent"),
        ("m_LineSpacing", "line_spacing"),
    ):
        value = metadata.get(key)
        if value is not None and hasattr(font, attr):
            setattr(font, attr, float(value))
    return metadata


def resolve_charset_source(
    charset_source: str | None,
    script_dir: str | None = None,
) -> str:
    """KR: TTF->SDF 생성에 사용할 글자셋 인자를 파일 경로 또는 리터럴 문자열로 정규화합니다.
    EN: Normalize the charset argument for TTF-to-SDF generation as a path or literal text.
    """
    if script_dir is None:
        script_dir = get_script_dir()
    if not charset_source:
        return os.path.join(script_dir, "CharList_3911.txt")

    raw = strip_wrapping_quotes_repeated(str(charset_source)).strip()
    if not raw:
        return os.path.join(script_dir, "CharList_3911.txt")

    candidates = [raw]
    if not os.path.isabs(raw):
        candidates.extend(
            [
                os.path.join(os.getcwd(), raw),
                os.path.join(script_dir, raw),
            ]
        )
    for candidate in _dedupe_preserve_order_str(candidates):
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return raw


@lru_cache(maxsize=1)
def _load_generated_font_assets_cached(
    script_dir: str,
    ttf_path: str,
    prefer_raster: bool = False,
    atlas_padding: int = 7,
    charset_source: str | None = None,
) -> JsonDict:
    try:
        import make_sdf as make_sdf_module
    except Exception as e:
        _log_warning(f"[make_sdf] import failed: {e!r}")
        return {}

    try:
        with open(ttf_path, "rb") as f:
            ttf_data = f.read()
    except Exception as e:
        _log_warning(f"[make_sdf] failed to read TTF: {ttf_path} ({e!r})")
        return {}

    try:
        charset_text = make_sdf_module._load_charset_text(
            resolve_charset_source(charset_source, script_dir)
        )
        unicodes = make_sdf_module._text_to_unicodes(charset_text)
    except Exception as e:
        _log_warning(f"[make_sdf] failed to load charset: {e!r}")
        return {}

    if not unicodes:
        _log_warning("[make_sdf] charset is empty.")
        return {}

    padding = max(1, int(atlas_padding or 7))
    render_mode = "raster" if prefer_raster else "sdf"
    generated = make_sdf_module.generate_sdf_assets_from_ttf(
        ttf_data=ttf_data,
        font_name=os.path.splitext(os.path.basename(ttf_path))[0],
        unicodes=unicodes,
        point_size=0,
        atlas_padding=padding,
        atlas_width=4096,
        atlas_height=4096,
        render_mode=render_mode,
        log_fn=_log_debug,
    )
    if not isinstance(generated, dict):
        _log_warning(f"[make_sdf] generation failed: {ttf_path}")
        return {}

    sdf_data = generated.get("sdf_data")
    sdf_data_normalized = generated.get("sdf_data_normalized")
    if isinstance(sdf_data, dict) and not isinstance(sdf_data_normalized, dict):
        sdf_data_normalized = normalize_sdf_data(sdf_data, deep_copy=True)

    generated_atlas_path = None
    generated_atlas = generated.get("sdf_atlas")
    if isinstance(generated_atlas, Image.Image):
        atlas_temp_dir = tempfile.mkdtemp(prefix="unity_font_replacer_generated_")
        register_temp_dir_for_cleanup(atlas_temp_dir)
        generated_atlas_path = os.path.join(atlas_temp_dir, "atlas.png")
        try:
            generated_atlas.save(generated_atlas_path, format="PNG")
        except Exception:
            shutil.rmtree(atlas_temp_dir, ignore_errors=True)
            generated_atlas_path = None
        finally:
            try:
                generated_atlas.close()
            except Exception:
                pass

    return {
        "ttf_data": generated.get("ttf_data") or ttf_data,
        "sdf_data": sdf_data,
        "sdf_data_normalized": sdf_data_normalized,
        "sdf_atlas_path": generated_atlas_path,
        "sdf_materials": generated.get("sdf_materials"),
        "sdf_swizzle": False,
        "sdf_process_swizzle": False,
        "padding_variant": padding,
    }


def _find_replacement_sdf_atlas_path(
    script_dir: str,
    normalized: str,
    prefer_raster: bool = False,
) -> str | None:
    kr_assets = os.path.join(script_dir, "KR_ASSETS")
    _, name_candidates = _build_font_asset_name_candidates(
        normalized, bool(prefer_raster)
    )
    for name_candidate in name_candidates:
        atlas_path = os.path.join(kr_assets, f"{name_candidate} Atlas.png")
        if os.path.exists(atlas_path):
            return atlas_path
    return None


@lru_cache(maxsize=128)
def _estimate_replacement_sdf_texture_bytes(
    script_dir: str,
    normalized: str,
    prefer_raster: bool = False,
) -> int:
    atlas_path = _find_replacement_sdf_atlas_path(
        script_dir,
        normalized,
        bool(prefer_raster),
    )
    if not atlas_path:
        return 0

    try:
        with Image.open(atlas_path) as atlas_image:
            width = int(atlas_image.width)
            height = int(atlas_image.height)
            try:
                channel_count = max(1, len(atlas_image.getbands()))
            except Exception:
                channel_count = 4
        if width <= 0 or height <= 0:
            return 0
        return width * height * channel_count
    except Exception:
        return 0


def _estimate_sdf_texture_batch_profile(
    file_sdf_replacements: dict[str, JsonDict],
    *,
    force_raster: bool = False,
    script_dir: str | None = None,
    batch_target_bytes: int = _AUTO_SPLIT_TEXTURE_BATCH_TARGET_BYTES,
) -> JsonDict:
    if script_dir is None:
        script_dir = get_script_dir()

    estimated_total_bytes = 0
    estimated_target_count = 0
    max_target_bytes = 0

    for info in file_sdf_replacements.values():
        if not isinstance(info, dict):
            continue
        replacement_font = str(info.get("Replace_to") or "").strip()
        if not replacement_font:
            continue
        prefer_raster = bool(force_raster) or parse_bool_flag(info.get("force_raster"))
        estimated_bytes = _estimate_replacement_sdf_texture_bytes(
            script_dir,
            normalize_font_name(replacement_font),
            prefer_raster,
        )
        if estimated_bytes <= 0:
            continue
        estimated_target_count += 1
        estimated_total_bytes += estimated_bytes
        max_target_bytes = max(max_target_bytes, estimated_bytes)

    suggested_batch_size = 0
    if estimated_target_count > 0 and max_target_bytes > 0:
        safe_target = max(1, int(batch_target_bytes or 0))
        suggested_batch_size = max(1, safe_target // max_target_bytes)
        suggested_batch_size = min(estimated_target_count, suggested_batch_size)

    return {
        "estimated_target_count": estimated_target_count,
        "estimated_total_bytes": estimated_total_bytes,
        "max_target_bytes": max_target_bytes,
        "suggested_batch_size": suggested_batch_size,
    }


@lru_cache(maxsize=2)
def _load_font_assets_cached(
    script_dir: str,
    normalized: str,
    prefer_raster: bool = False,
    padding_variant: int | None = None,
) -> JsonDict:
    """KR: KR_ASSETS에서 폰트 리소스를 읽어 캐시에 저장합니다.
    EN: Reads font resources from KR_ASSETS and stores them in cache.
    """
    kr_assets = os.path.join(script_dir, "KR_ASSETS")
    asset_roots = _iter_kr_asset_roots(kr_assets, padding_variant=padding_variant)
    font_name_candidates, name_candidates = _build_font_asset_name_candidates(
        normalized,
        bool(prefer_raster),
    )

    ttf_data = None
    for font_name in font_name_candidates:
        for ext in (".ttf", ".otf"):
            for asset_root in asset_roots:
                font_path = os.path.join(asset_root, f"{font_name}{ext}")
                if os.path.exists(font_path):
                    with open(font_path, "rb") as f:
                        ttf_data = f.read()
                    break
            if ttf_data is not None:
                break
        if ttf_data is not None:
            break

    sdf_data = None
    sdf_data_normalized = None
    sdf_swizzle = False
    sdf_process_swizzle = False
    for name_candidate in name_candidates:
        for asset_root in asset_roots:
            sdf_json_path = os.path.join(asset_root, f"{name_candidate}.json")
            if not os.path.exists(sdf_json_path):
                continue
            with open(sdf_json_path, "r", encoding="utf-8") as f:
                sdf_data = json.load(f)
            if isinstance(sdf_data, dict):
                sdf_data_normalized = normalize_sdf_data(sdf_data, deep_copy=True)
                sdf_swizzle = parse_bool_flag(sdf_data.get("swizzle"))
                sdf_process_swizzle = parse_bool_flag(sdf_data.get("process_swizzle"))
            break
        if sdf_data is not None:
            break

    sdf_atlas_path = None
    for name_candidate in name_candidates:
        for asset_root in asset_roots:
            candidate_atlas_path = os.path.join(
                asset_root, f"{name_candidate} Atlas.png"
            )
            if not os.path.exists(candidate_atlas_path):
                continue
            sdf_atlas_path = candidate_atlas_path
            break
        if sdf_atlas_path is not None:
            break

    sdf_material_data = None
    for name_candidate in name_candidates:
        for asset_root in asset_roots:
            sdf_material_path = os.path.join(
                asset_root, f"{name_candidate} Material.json"
            )
            if not os.path.exists(sdf_material_path):
                continue
            with open(sdf_material_path, "r", encoding="utf-8") as f:
                sdf_material_data = json.load(f)
            break
        if sdf_material_data is not None:
            break

    return {
        "ttf_data": ttf_data,
        "sdf_data": sdf_data,
        "sdf_data_normalized": sdf_data_normalized,
        # KR: 디코드된 4096x4096 PIL 이미지는 캐시하지 않고 경로만 보관합니다.
        # EN: Cache only the path, not a decoded 4096x4096 PIL image.
        "sdf_atlas_path": sdf_atlas_path,
        "sdf_materials": sdf_material_data,
        "sdf_swizzle": sdf_swizzle,
        "sdf_process_swizzle": sdf_process_swizzle,
        "padding_variant": int(padding_variant) if padding_variant is not None else None,
    }


def load_font_assets(
    font_name: str,
    prefer_raster: bool = False,
    padding_variant: int | None = None,
    generate_sdf: bool = True,
    charset_source: str | None = None,
) -> JsonDict:
    """KR: 지정 폰트명의 교체용 리소스(TTF/SDF/Atlas/Material)를 로드합니다.
    EN: Loads replacement resources (TTF/SDF/Atlas/Material) for the specified font name.
    """
    script_dir = get_script_dir()
    source = strip_wrapping_quotes_repeated(str(font_name))
    normalized = normalize_font_name(source)
    cached_assets = _load_font_assets_cached(
        script_dir,
        normalized,
        bool(prefer_raster),
        int(padding_variant) if padding_variant is not None else None,
    )
    ttf_path = _resolve_ttf_source_path(source, script_dir)
    explicit_ttf_source = (
        ttf_path is not None
        and os.path.splitext(source)[1].lower() in {".ttf", ".otf"}
    )
    ttf_data = cached_assets["ttf_data"]
    if ttf_data is None and ttf_path:
        try:
            with open(ttf_path, "rb") as f:
                ttf_data = f.read()
        except Exception:
            ttf_data = None

    generated_assets: JsonDict = {}
    if (
        generate_sdf
        and ttf_path
        and (
            explicit_ttf_source
            or charset_source
            or not (
                cached_assets.get("sdf_data")
                and cached_assets.get("sdf_atlas_path")
            )
        )
    ):
        generated_assets = _load_generated_font_assets_cached(
            script_dir,
            ttf_path,
            bool(prefer_raster),
            int(padding_variant) if padding_variant is not None else 7,
            resolve_charset_source(charset_source, script_dir),
        )

    generated_atlas = None
    if generated_assets.get("sdf_data") and generated_assets.get("sdf_atlas_path"):
        generated_atlas_path = generated_assets.get("sdf_atlas_path")
        if isinstance(generated_atlas_path, str) and generated_atlas_path:
            try:
                with Image.open(generated_atlas_path) as source_atlas:
                    generated_atlas = source_atlas.copy()
                    generated_atlas.load()
            except Exception:
                generated_atlas = None
        if generated_atlas is None:
            generated_assets = {}

    if generated_assets.get("sdf_data") and isinstance(
        generated_atlas, Image.Image
    ):
        return {
            "ttf_data": generated_assets.get("ttf_data") or ttf_data,
            "sdf_data": generated_assets.get("sdf_data"),
            "sdf_data_normalized": generated_assets.get("sdf_data_normalized"),
            "sdf_atlas": generated_atlas,
            "sdf_materials": generated_assets.get("sdf_materials"),
            "sdf_swizzle": generated_assets.get("sdf_swizzle"),
            "sdf_process_swizzle": bool(
                generated_assets.get("sdf_process_swizzle", False)
            ),
            "padding_variant": generated_assets.get("padding_variant"),
        }

    atlas = None
    atlas_path = cached_assets.get("sdf_atlas_path")
    if isinstance(atlas_path, str) and atlas_path:
        try:
            with Image.open(atlas_path) as source_atlas:
                atlas = source_atlas.copy()
                atlas.load()
        except Exception:
            atlas = None
    return {
        "ttf_data": ttf_data,
        "sdf_data": cached_assets["sdf_data"],
        "sdf_data_normalized": cached_assets.get("sdf_data_normalized"),
        # KR: 캐시된 atlas 객체를 재사용하여 교체 시 이미지 중복 생성을 방지합니다.
    # EN: Reuses cached atlas objects to prevent duplicate image creation during replacement.
        "sdf_atlas": atlas,
        "sdf_materials": cached_assets["sdf_materials"],
        "sdf_swizzle": cached_assets.get("sdf_swizzle"),
        "sdf_process_swizzle": bool(cached_assets.get("sdf_process_swizzle", False)),
        "padding_variant": cached_assets.get("padding_variant"),
    }


# KR: TypeTree에 정의되지 않은 trailing bytes를 ObjectReader path_id 기준으로 보존합니다.
# EN: Preserves trailing bytes not defined in TypeTree, keyed by ObjectReader path_id.
_trailing_bytes_store: dict[int, tuple[weakref.ReferenceType[Any], bytes]] = {}


def _store_trailing_bytes(obj: Any, trailing: bytes) -> None:
    obj_id = id(obj)
    if trailing:
        def _discard_dead(ref: weakref.ReferenceType[Any], key: int = obj_id) -> None:
            current = _trailing_bytes_store.get(key)
            if current is not None and current[0] is ref:
                _trailing_bytes_store.pop(key, None)

        _trailing_bytes_store[obj_id] = (
            weakref.ref(obj, _discard_dead),
            bytes(trailing),
        )
    else:
        _trailing_bytes_store.pop(obj_id, None)


def _pop_trailing_bytes(obj: Any) -> bytes:
    entry = _trailing_bytes_store.pop(id(obj), None)
    if entry is None:
        return b""
    obj_ref, trailing = entry
    return trailing if obj_ref() is obj else b""


class _TrailingBytesReplacer:
    """Stream a UnityPy Replacer followed by preserved unknown bytes."""

    __slots__ = ("base", "suffix", "size")

    def __init__(self, base: Replacer, suffix: bytes):
        self.base = base
        self.suffix = bytes(suffix)
        self.size = len(base) + len(self.suffix)

    def __len__(self) -> int:
        return self.size

    def iter_chunks(self, chunk_size: int = 1024 * 1024):
        yield from self.base.iter_chunks(chunk_size)
        if self.suffix:
            yield self.suffix

    def write_to(self, writer: Any, chunk_size: int = 1024 * 1024) -> None:
        self.base.write_to(writer, chunk_size)
        if self.suffix:
            writer.write(self.suffix)

    def read_bytes(self) -> bytes:
        return self.base.read_bytes() + self.suffix

    def cleanup(self) -> None:
        self.base.cleanup()


class _SegmentedBytesReplacer:
    """Stream byte segments without joining a full replacement object in memory."""

    __slots__ = ("segments", "size")

    def __init__(self, segments: Iterable[bytes | bytearray | memoryview]):
        self.segments = [segment for segment in segments if len(segment)]
        self.size = sum(len(segment) for segment in self.segments)

    def __len__(self) -> int:
        return self.size

    def iter_chunks(self, chunk_size: int = 1024 * 1024):
        for segment in self.segments:
            view = segment if isinstance(segment, memoryview) else memoryview(segment)
            position = 0
            while position < len(view):
                next_position = min(position + chunk_size, len(view))
                yield view[position:next_position]
                position = next_position

    def write_to(self, writer: Any, chunk_size: int = 1024 * 1024) -> None:
        for chunk in self.iter_chunks(chunk_size):
            writer.write(chunk)

    def read_bytes(self) -> bytes:
        return b"".join(bytes(segment) for segment in self.segments)

    def cleanup(self) -> None:
        for segment in self.segments:
            if isinstance(segment, memoryview):
                try:
                    segment.release()
                except Exception:
                    pass
        self.segments.clear()


def _append_trailing_bytes(obj: Any, trailing: bytes) -> None:
    """Append trailing bytes without materializing a file-backed Replacer."""
    if not trailing:
        return
    current = getattr(obj, "data", None)
    if isinstance(current, Replacer):
        # Avoid ObjectReader.set_raw_data() cleaning the base replacer that the
        # composite still needs. Environment cleanup releases it after saving.
        obj.data = _TrailingBytesReplacer(current, trailing)
        obj.assets_file.mark_changed()
        return
    current_data = obj.get_raw_data()
    obj.set_raw_data(current_data + trailing)


def _capture_trailing_bytes(obj: Any) -> bytes:
    """KR: TypeTree 파싱 후 읽히지 않은 trailing bytes를 캡처합니다.
    EN: Captures unread trailing bytes after TypeTree parsing.
    """
    pos = obj.reader.Position
    end = obj.byte_start + obj.byte_size
    if pos < end:
        remaining = obj.reader.read_bytes(end - pos)
        obj.reader.Position = pos
        return remaining
    return b""


def _safe_parse_as_object(obj: Any, **kwargs: Any) -> Any:
    """KR: parse_as_object()를 check_read=True로 먼저 시도하고,
    바이트 크기 불일치(중국판 Unity 등)로 실패하면 check_read=False로 재시도하고
    trailing bytes를 별도 저장소에 보존합니다.
    EN: Tries parse_as_object() with check_read=True first.
    On byte size mismatch (e.g. China Unity), retries with check_read=False
    and preserves trailing bytes in a separate store.
    """
    obj_id = id(obj)
    try:
        result = obj.parse_as_object(check_read=True, **kwargs)
        _trailing_bytes_store.pop(obj_id, None)
        return result
    except ValueError as e:
        if "Expected to read" in str(e) and "bytes" in str(e):
            result = obj.parse_as_object(check_read=False, **kwargs)
            trailing = _capture_trailing_bytes(obj)
            _store_trailing_bytes(obj, trailing)
            return result
        raise


def _safe_parse_as_dict(obj: Any, **kwargs: Any) -> dict[str, Any]:
    """KR: parse_as_dict()를 check_read=True로 먼저 시도하고,
    바이트 크기 불일치로 실패하면 check_read=False로 재시도하고
    trailing bytes를 별도 저장소에 보존합니다.
    EN: Tries parse_as_dict() with check_read=True first.
    On byte size mismatch, retries with check_read=False
    and preserves trailing bytes in a separate store.
    """
    obj_id = id(obj)
    try:
        result = obj.parse_as_dict(check_read=True, **kwargs)
        _trailing_bytes_store.pop(obj_id, None)
        return result
    except ValueError as e:
        if "Expected to read" in str(e) and "bytes" in str(e):
            result = obj.parse_as_dict(check_read=False, **kwargs)
            trailing = _capture_trailing_bytes(obj)
            _store_trailing_bytes(obj, trailing)
            return result
        raise


def _safe_save(obj: Any, parse_dict: Any) -> None:
    """KR: save() 후 trailing bytes가 있으면 raw data에 append합니다.
    EN: After save(), appends trailing bytes to raw data if present.
    """
    parse_dict.save()
    trailing = _pop_trailing_bytes(obj)
    _append_trailing_bytes(obj, trailing)


def _has_trailing_bytes(obj: Any) -> bool:
    """KR: 이 오브젝트에 TypeTree로 읽히지 않는 trailing bytes가 있는지 확인합니다.
    EN: Checks whether this object has trailing bytes not read by TypeTree.
    """
    entry = _trailing_bytes_store.get(id(obj))
    return entry is not None and entry[0]() is obj


class _CountingWriteStream:
    """Seekable write sink that tracks length without retaining written bytes."""

    def __init__(self) -> None:
        self._position = 0
        self._length = 0
        self.closed = False

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            position = int(offset)
        elif whence == 1:
            position = self._position + int(offset)
        elif whence == 2:
            position = self._length + int(offset)
        else:
            raise ValueError(f"invalid whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        return b""

    def write(self, data: Any) -> int:
        size = len(data)
        self._position += size
        self._length = max(self._length, self._position)
        return size

    def close(self) -> None:
        self.closed = True


def _detect_typetree_size_mismatch(obj: Any, parsed_value: Any | None = None) -> bool:
    """Detect China-Unity-style fields omitted by TypeTree serialization.

    The previous implementation serialized into an in-memory ``BytesIO`` and
    re-read the full Texture2D.  Count the serialized bytes instead, and let
    callers pass the object they already parsed so large atlas data is neither
    duplicated nor parsed twice.
    """
    writer = None
    try:
        from UnityPy.helpers.TypeTreeHelper import write_typetree
        from UnityPy.streams import EndianBinaryWriter

        value = (
            parsed_value
            if parsed_value is not None
            else obj.read_typetree(check_read=False)
        )
        node = obj._get_typetree_node()
        writer = EndianBinaryWriter(
            _CountingWriteStream(),
            endian=obj.reader.endian,
        )
        write_typetree(value, node, writer, obj.assets_file)
        return int(writer.Length) < int(obj.byte_size)
    except Exception:
        return False
    finally:
        if writer is not None:
            try:
                writer.dispose()
            except Exception:
                pass


def _binary_patch_texture2d(
    obj: Any,
    *,
    image_data: bytes,
    width: int,
    height: int,
    lang: str = "ko",
) -> bool:
    """KR: Texture2D를 TypeTree 재직렬화 없이 바이너리 패치합니다.
    중국판 Unity 등에서 TypeTree가 커버하지 못하는 extra bytes가 있을 때 사용합니다.
    EN: Binary-patches Texture2D without TypeTree re-serialization.
    Used when extra bytes not covered by TypeTree exist (e.g. China Unity).
    """
    import struct as _struct

    original_raw = obj.get_raw_data()
    if len(original_raw) < 48:
        return False

    # KR: 원본 raw에서 스트림 경로 문자열을 찾아 필드 위치를 역추적합니다.
    # 스트리밍 경로 문자열 검색 (.resS 또는 .resource)
    # EN: Finds stream path strings in the original raw data to trace back field positions.
    # Searches for streaming path strings (.resS or .resource)
    stream_path_marker = None
    for marker in [b".resS", b".resource"]:
        idx = original_raw.find(marker)
        if idx >= 0:
            # KR: 문자열 시작 위치를 찾기 위해 앞쪽으로 탐색
            # EN: Scan backwards to find the start position of the string
            str_start = idx
            while str_start > 0 and original_raw[str_start - 1:str_start] not in (b"\x00",):
                str_start -= 1
                if idx - str_start > 200:
                    break
            # KR: string length prefix는 str_start - 4 위치
            # EN: The string length prefix is at position str_start - 4
            path_len_pos = str_start - 4
            if path_len_pos < 0:
                continue
            try:
                path_len = _struct.unpack_from("<i", original_raw, path_len_pos)[0]
                if 0 < path_len < 256 and path_len_pos + 4 + path_len <= len(original_raw):
                    stream_path_marker = (path_len_pos, path_len, str_start)
                    break
            except Exception:
                continue

    if stream_path_marker is not None:
        # KR: 스트리밍 모드 — 경로 문자열 기준으로 필드 위치를 역추적합니다.
        # EN: Streaming mode -- traces back field positions based on the path string.
        path_len_pos, path_len, path_str_start = stream_path_marker
        stream_size_pos = path_len_pos - 4
        stream_offset_pos = stream_size_pos - 8
        image_data_size_pos = stream_offset_pos - 4
        orig_stream_end = path_str_start + path_len
        orig_stream_end += (4 - orig_stream_end % 4) % 4
    else:
        image_data_size_pos = -1
        orig_stream_end = len(original_raw)

    # KR: TypeTree 파싱으로 정확한 필드 오프셋을 구하고, 원본 raw를 직접 패치합니다.
    # EN: Obtains exact field offsets via TypeTree parsing and directly patches the original raw data.
    from UnityPy.helpers.TypeTreeHelper import TypeTreeConfig as _TTC, read_value as _rv
    from UnityPy.streams import EndianBinaryReader as _EBR
    field_offsets: dict[str, int] = {}
    typetree_end: int | None = None
    _tmp_reader = None
    _root_node = None
    try:
        _tmp_reader = _EBR(original_raw, endian=obj.reader.endian)
        _tmp_config = _TTC(True, obj.assets_file, False)
        _root_node = obj._get_typetree_node()
        for _child in _root_node.m_Children:
            _pos_before = _tmp_reader.Position
            field_offsets[_child.m_Name] = _pos_before
            if _child.m_Name == "image data" and _child.m_Type == "TypelessData":
                _image_length = int(_tmp_reader.read_int())
                if _image_length < 0 or _image_length > (
                    _tmp_reader.Length - _tmp_reader.Position
                ):
                    raise ValueError("invalid Texture2D image data length")
                _tmp_reader.Position += _image_length
                _tmp_reader.align_stream()
            else:
                _rv(_child, _tmp_reader, _tmp_config)
        typetree_end = int(_tmp_reader.Position)
    except Exception:
        pass
    finally:
        if _tmp_reader is not None:
            try:
                _tmp_reader.dispose()
            except Exception:
                pass

    # KR: image data 필드의 시작 오프셋 = image_data_size_pos (TypeTree 기준)
    # EN: Start offset of the image data field = image_data_size_pos (TypeTree basis)
    if "image data" in field_offsets:
        image_data_size_pos = field_offsets["image data"]

    if (
        stream_path_marker is None
        and (image_data_size_pos < 0 or typetree_end is None)
    ):
        # Rare fallback for TypeTrees whose child traversal failed. This path
        # may materialize the old image once, but the normal binary fallback
        # stays segmented and streaming.
        try:
            d_temp = obj.read_typetree(check_read=False)
            orig_img_data = d_temp.get("image data", b"")
            orig_img_len = (
                len(orig_img_data)
                if isinstance(orig_img_data, (bytes, bytearray, memoryview))
                else 0
            )
            obj.reset()
            pos0 = obj.reader.Position
            obj.read_typetree(check_read=False)
            typetree_bytes = obj.reader.Position - pos0
            trailing_size = len(original_raw) - typetree_bytes
            img_block_size = 4 + orig_img_len
            img_block_padded = img_block_size + (4 - img_block_size % 4) % 4
            if image_data_size_pos < 0:
                image_data_size_pos = (
                    len(original_raw) - trailing_size - 16 - img_block_padded
                )
            orig_stream_end = len(original_raw) - trailing_size
            del orig_img_data, d_temp
        except Exception:
            return False

    if image_data_size_pos < 0 or image_data_size_pos >= len(original_raw):
        return False

    part4_start = (
        typetree_end
        if "image data" in field_offsets and typetree_end is not None
        else orig_stream_end
    )
    if part4_start < image_data_size_pos or part4_start > len(original_raw):
        return False

    part1 = bytearray(memoryview(original_raw)[:image_data_size_pos])

    # KR: 정확한 오프셋으로 필드 패치 (패턴 검색 대신 직접 오프셋 사용)
    # EN: Patches fields at exact offsets (uses direct offsets instead of pattern search)
    if "m_Width" in field_offsets and field_offsets["m_Width"] + 4 <= len(part1):
        _struct.pack_into("<i", part1, field_offsets["m_Width"], width)
    if "m_Height" in field_offsets and field_offsets["m_Height"] + 4 <= len(part1):
        _struct.pack_into("<i", part1, field_offsets["m_Height"], height)
    if "m_CompleteImageSize" in field_offsets and field_offsets["m_CompleteImageSize"] + 4 <= len(part1):
        _struct.pack_into("<I", part1, field_offsets["m_CompleteImageSize"], len(image_data))

    image_size_prefix = _struct.pack("<i", len(image_data))
    image_block_size = len(image_size_prefix) + len(image_data)
    image_padding = b"\x00" * ((4 - image_block_size % 4) % 4)
    empty_stream_data = b""
    stream_node = next(
        (
            child
            for child in getattr(_root_node, "m_Children", [])
            if child.m_Name == "m_StreamData"
        ),
        None,
    )
    if stream_node is not None:
        from UnityPy.helpers.TypeTreeHelper import write_typetree as _write_typetree
        from UnityPy.streams import EndianBinaryWriter as _EBW

        stream_writer = _EBW(endian=obj.reader.endian)
        try:
            empty_stream_value = {
                child.m_Name: "" if child.m_Type == "string" else 0
                for child in stream_node.m_Children
            }
            _write_typetree(
                empty_stream_value,
                stream_node,
                stream_writer,
                obj.assets_file,
            )
            empty_stream_data = stream_writer.bytes
        except Exception:
            return False
        finally:
            stream_writer.dispose()
    # Copy only the normally-small unknown tail so the segmented replacer does
    # not keep the entire old Texture2D payload alive through a memoryview.
    trailing_bytes = bytes(memoryview(original_raw)[part4_start:])
    replacement = _SegmentedBytesReplacer(
        (
            part1,
            image_size_prefix,
            image_data,
            image_padding,
            empty_stream_data,
            trailing_bytes,
        )
    )
    obj.set_raw_data(replacement)

    if lang == "ko":
        _log_debug(
            f"[binary_patch_texture2d] PathID={obj.path_id} "
            f"orig_raw={len(original_raw)}B new_raw={len(replacement):,}B "
            f"trailing={len(trailing_bytes)}B"
        )
    return True


def _cleanup_replace_call_resources(func: Callable[..., bool]) -> Callable[..., bool]:
    """Remove per-call save and newly spilled payload files on exceptions."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> bool:
        tmp_path: str | None = None
        deferred_payload_dir: str | None = None
        created_payload_dir = False
        try:
            bound = inspect.signature(func).bind_partial(*args, **kwargs)
            game_path = str(bound.arguments.get("game_path", ""))
            temp_root_dir = bound.arguments.get("temp_root_dir")
            tmp_root = (
                os.path.abspath(str(temp_root_dir))
                if temp_root_dir is not None
                else os.path.join(get_data_path(game_path), "temp")
            )
            tmp_path = os.path.join(tmp_root, "unity_font_replacer_temp")
            supplied_payload_dir = bound.arguments.get("_deferred_payload_dir")
            if supplied_payload_dir:
                deferred_payload_dir = os.path.abspath(str(supplied_payload_dir))
            else:
                deferred_payload_root = os.path.join(
                    tmp_root, "deferred_patch_payloads"
                )
                os.makedirs(deferred_payload_root, exist_ok=True)
                deferred_payload_dir = tempfile.mkdtemp(
                    prefix="call_", dir=deferred_payload_root
                )
                kwargs["_deferred_payload_dir"] = deferred_payload_dir
                created_payload_dir = True
        except Exception:
            tmp_path = None
        try:
            return func(*args, **kwargs)
        except BaseException:
            if created_payload_dir and deferred_payload_dir:
                shutil.rmtree(deferred_payload_dir, ignore_errors=True)
            raise
        finally:
            if tmp_path and os.path.isdir(tmp_path):
                try:
                    shutil.rmtree(tmp_path)
                except Exception:
                    pass

    return wrapper


@cleanup_unitypy_environments
@_cleanup_replace_call_resources
def replace_fonts_in_file(
    unity_version: str,
    game_path: str,
    assets_file: str,
    replacements: dict[str, JsonDict],
    replace_ttf: bool = True,
    replace_sdf: bool = True,
    use_game_mat: bool = False,
    use_game_line_metrics: bool = False,
    force_raster: bool = False,
    material_scale_by_padding: bool = True,
    outline_ratio: float = 1.0,
    prefer_original_compress: bool = False,
    temp_root_dir: str | None = None,
    generator: TypeTreeGenerator | None = None,
    replacement_lookup: dict[tuple[str, str, str, int], str] | None = None,
    ps5_swizzle: bool = False,
    preview_export: bool = False,
    preview_root: str | None = None,
    prefer_builtin_padding_variants: bool = False,
    charset_source: str | None = None,
    asset_file_index: dict[str, Any] | None = None,
    deferred_texture_plans: dict[str, dict[str, Any]] | None = None,
    deferred_material_plans: dict[str, dict[str, Any]] | None = None,
    deferred_material_atlas_plans: dict[str, dict[str, Any]] | None = None,
    collected_material_atlas_plans: dict[str, JsonDict] | None = None,
    pending_external_patch_files: set[str] | None = None,
    logical_file_key: str | None = None,
    phase_callback: Callable[[str, JsonDict], None] | None = None,
    lang: Language = "ko",
    deferred_transaction: _DeferredPatchTransaction | None = None,
    operation_outcome: JsonDict | None = None,
    _deferred_payload_dir: str | None = None,
) -> bool:
    """KR: 단일 assets 파일의 TTF/SDF 폰트를 교체하고 저장합니다.

    기본 모드는 줄 간격 관련 메트릭(LineHeight/Ascender/Descender 등)을 게임 원본 비율로 보정해
    교체 pointSize에 맞춰 적용합니다.
    use_game_line_metrics=True면 게임 원본 줄 간격 메트릭을 그대로 사용합니다.
    pointSize는 옵션과 무관하게 교체 폰트 값을 유지합니다.
    material_scale_by_padding=True면 SDF 머티리얼 float를 (게임 padding / 교체 padding) 비율로 보정합니다.
    outline_ratio는 현재 선택된 Material 기준(_OutlineWidth/_OutlineSoftness)에 배율로 적용합니다.
    prefer_original_compress=True면 원본 압축 우선, False면 무압축 계열 우선 저장 전략을 사용합니다.
    ps5_swizzle=True면 대상 Atlas의 swizzle 상태를 판별해 교체 Atlas를 자동 swizzle/unswizzle합니다.
    preview_export=True면 preview 폴더에 Atlas/Glyph crop 미리보기를 저장합니다.
    ps5_swizzle=True일 때는 unswizzle 기준으로 저장합니다.
    temp_root_dir가 지정되면 임시 저장 디렉터리 루트로 사용합니다.
    EN: Replaces TTF/SDF fonts in a single assets file and saves it.

    Default mode adjusts line-spacing metrics (LineHeight/Ascender/Descender etc.) by the game's original ratio
    and applies them scaled to the replacement pointSize.
    use_game_line_metrics=True uses the game's original line-spacing metrics as-is.
    pointSize always retains the replacement font's value regardless of options.
    material_scale_by_padding=True adjusts SDF material floats by (game padding / replacement padding) ratio.
    outline_ratio applies as a multiplier to the selected Material's _OutlineWidth/_OutlineSoftness.
    prefer_original_compress=True uses original compression first; False uses uncompressed-first strategy.
    ps5_swizzle=True detects target Atlas swizzle state and auto-swizzles/unswizzles the replacement Atlas.
    preview_export=True saves Atlas/Glyph crop previews to the preview folder.
    When ps5_swizzle=True, saves based on unswizzle.
    temp_root_dir, if specified, is used as the temp storage directory root.
    """
    fn_without_path = os.path.basename(assets_file)
    if operation_outcome is not None:
        operation_outcome.clear()
    current_file_key = _resolve_current_file_key(assets_file, logical_file_key)
    data_path = get_data_path(game_path, lang=lang)
    using_custom_temp_root = temp_root_dir is not None
    tmp_root = (
        os.path.abspath(temp_root_dir)
        if using_custom_temp_root
        else os.path.join(data_path, "temp")
    )
    tmp_path = os.path.join(tmp_root, "unity_font_replacer_temp")
    if using_custom_temp_root:
        register_temp_dir_for_cleanup(tmp_path)
    else:
        register_temp_dir_for_cleanup(tmp_root)
    bundle_signatures = BUNDLE_SIGNATURES
    source_bundle_signature = _read_bundle_signature(assets_file, bundle_signatures)

    if not os.path.exists(tmp_root):
        os.makedirs(tmp_root, exist_ok=True)

    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    os.makedirs(tmp_path, exist_ok=True)
    deferred_payload_dir = _deferred_payload_dir or os.path.join(
        tmp_root, "deferred_patch_payloads"
    )
    os.makedirs(deferred_payload_dir, exist_ok=True)
    register_temp_dir_for_cleanup(deferred_payload_dir)

    phase_started_at = time.perf_counter()
    _emit_phase_callback(
        phase_callback,
        "load_begin",
        file=fn_without_path,
        path=assets_file,
    )
    env = load_unitypy(assets_file)
    _emit_phase_callback(
        phase_callback,
        "load_end",
        file=fn_without_path,
        elapsed_sec=(time.perf_counter() - phase_started_at),
    )
    env_file = getattr(env, "file", None)
    if env_file is None:
        files = getattr(env, "files", None)
        if isinstance(files, dict) and len(files) == 1:
            env_file = next(iter(files.values()))
    if env_file is None:
        raise RuntimeError(
            "Could not determine primary UnityPy file object for saving."
        )
    if not preview_export:
        _ensure_custom_unitypy_streaming_save(lang=lang)
    if generator is None and replace_sdf:
        compile_method = get_compile_method(data_path)
        generator = _create_generator(
            unity_version, game_path, data_path, compile_method, lang=lang
        )
    env.typetree_generator = generator
    if replacement_lookup is None:
        replacement_lookup, _ = build_replacement_lookup(replacements)
    replacement_meta_lookup: dict[tuple[str, str, str, int], JsonDict] = {}
    preview_target_lookup: dict[tuple[str, str, int], JsonDict] = {}
    for info in replacements.values():
        if not isinstance(info, dict):
            continue
        type_raw = info.get("Type")
        file_raw = info.get("File")
        assets_raw = info.get("assets_name")
        path_raw = info.get("Path_ID")
        if (
            not isinstance(type_raw, str)
            or not isinstance(file_raw, str)
            or not isinstance(assets_raw, str)
        ):
            continue
        try:
            path_id = int(path_raw)
        except (TypeError, ValueError):
            continue
        if type_raw == "SDF":
            preview_target_lookup[(file_raw, assets_raw, path_id)] = info
        if not info.get("Replace_to"):
            continue
        replacement_meta_lookup[(type_raw, file_raw, assets_raw, path_id)] = info

    texture_object_lookup: dict[tuple[str, int], Any] = {}
    texture_swizzle_state_cache: dict[str, tuple[str | None, str | None]] = {}
    material_object_count_by_pathid: dict[int, int] = {}
    for item in env.objects:
        item_type = item.type.name
        if item_type == "Texture2D":
            texture_object_lookup[(item.assets_file.name, int(item.path_id))] = item
            continue
        if item_type == "Material":
            material_path_id = int(item.path_id)
            material_object_count_by_pathid[material_path_id] = (
                material_object_count_by_pathid.get(material_path_id, 0) + 1
            )

    target_ttf_targets: set[tuple[str, int]] = set()
    satisfied_ttf_targets: set[tuple[str, int]] = set()
    if replace_ttf:
        for key in replacement_lookup:
            if len(key) == 4 and key[0] == "TTF" and key[1] == fn_without_path:
                target_ttf_targets.add((str(key[2]), int(key[3])))

    target_sdf_targets: set[tuple[str, int]] = set()
    replacement_sdf_targets: set[tuple[str, int]] = set()
    target_sdf_pathids: set[int] = set()
    target_sdf_font_by_target: dict[tuple[str, int], str] = {}
    old_line_metric_keys = _OLD_LINE_METRIC_KEYS
    old_line_metric_scale_keys = _OLD_LINE_METRIC_SCALE_KEYS
    new_line_metric_keys = _NEW_LINE_METRIC_KEYS
    new_line_metric_scale_keys = _NEW_LINE_METRIC_SCALE_KEYS
    material_padding_scale_keys = _MATERIAL_PADDING_SCALE_KEYS
    replacement_padding_limit_warned: set[tuple[str, str, int]] = set()

    if replace_sdf:
        for key, value in replacement_lookup.items():
            if len(key) == 4 and key[0] == "SDF" and key[1] == fn_without_path:
                assets_key = key[2]
                path_id = key[3]
                target_key = (str(assets_key), int(path_id))
                target_sdf_targets.add(target_key)
                replacement_sdf_targets.add(target_key)
                target_sdf_pathids.add(path_id)
                target_sdf_font_by_target.setdefault(target_key, value)
        if preview_export:
            for file_name, assets_name, path_id in preview_target_lookup.keys():
                if file_name != fn_without_path:
                    continue
                target_key = (str(assets_name), int(path_id))
                target_sdf_targets.add(target_key)
                target_sdf_pathids.add(int(path_id))
    matched_sdf_targets = 0
    patched_sdf_targets = 0
    patched_sdf_target_keys: set[tuple[str, int]] = set()
    sdf_parse_failure_reasons: list[str] = []

    incoming_texture_plans = _copy_patch_bucket(
        deferred_texture_plans, current_file_key
    )
    incoming_material_plans = _copy_patch_bucket(
        deferred_material_plans, current_file_key
    )
    incoming_material_atlas_plans = _copy_patch_bucket(
        deferred_material_atlas_plans, current_file_key
    )
    incoming_texture_ids = _patch_payload_ids(incoming_texture_plans)
    incoming_material_ids = _patch_payload_ids(incoming_material_plans)
    incoming_material_atlas_ids = _patch_payload_ids(incoming_material_atlas_plans)
    consumed_texture_ids: set[int] = set()
    consumed_material_ids: set[int] = set()
    handled_material_atlas_ids: set[int] = set()
    required_local_texture_ids: set[int] = set()
    required_local_material_ids: set[int] = set()
    required_resolution_errors: list[str] = []

    texture_patch_plans: dict[str, Any] = dict(incoming_texture_plans)
    owned_texture_patch_plans: dict[str, Any] = {}
    material_replacements: dict[str, JsonDict] = cast(
        dict[str, JsonDict],
        dict(incoming_material_plans),
    )
    material_replacements_by_pathid: dict[int, JsonDict] = {}
    material_replacements_by_atlas: dict[str, JsonDict] = cast(
        dict[str, JsonDict],
        dict(incoming_material_atlas_plans),
    )
    staged_texture_plans: dict[str, dict[str, Any]] = {}
    staged_material_plans: dict[str, dict[str, Any]] = {}
    staged_material_atlas_plans: dict[str, dict[str, Any]] = {}
    staged_pending_files: set[str] = set()
    ambiguous_material_fallback_warned: set[int] = set()
    modified = False
    save_success = False

    for obj in env.objects:
        assets_name = obj.assets_file.name
        if obj.type.name == "Font" and replace_ttf:
            font_pathid = obj.path_id
            replacement_font = replacement_lookup.get(
                ("TTF", fn_without_path, assets_name, font_pathid)
            )

            if replacement_font:
                assets = load_font_assets(replacement_font, generate_sdf=False)
                if assets["ttf_data"]:
                    font = _safe_parse_as_object(obj)
                    _raw_font_data = getattr(font, "m_FontData", b"")
                    current_ttf_data = _raw_font_data if isinstance(_raw_font_data, bytes) else bytes(_raw_font_data)
                    metadata_before = {
                        attr: copy.deepcopy(getattr(font, attr, None))
                        for attr in (
                            "m_FontNames",
                            "m_Ascent",
                            "m_Descent",
                            "m_LineSpacing",
                        )
                    }
                    metadata = apply_ttf_metadata_to_font(
                        font,
                        assets["ttf_data"],
                        fallback_name=replacement_font,
                    )
                    metadata_after = {
                        attr: copy.deepcopy(getattr(font, attr, None))
                        for attr in metadata_before
                    }
                    metadata_changed = metadata_before != metadata_after
                    same_font_data = current_ttf_data == assets["ttf_data"]
                    if same_font_data and not metadata_changed:
                        satisfied_ttf_targets.add((str(assets_name), int(font_pathid)))
                        _log_debug(
                            f"[replace_ttf] file={fn_without_path} assets={assets_name} path_id={font_pathid} "
                            f"name={font.m_Name} target={replacement_font} action=skip_same size={len(current_ttf_data)}"
                        )
                        if lang == "ko":
                            _log_console(
                                f"TTF 폰트 동일(건너뜀): {assets_name} | {font.m_Name} | "
                                f"(PathID: {font_pathid} == {replacement_font})"
                            )
                        else:
                            _log_console(
                                f"TTF already same (skip): {assets_name} | {font.m_Name} | "
                                f"(PathID: {font_pathid} == {replacement_font})"
                        )
                        continue
                    if lang == "ko" and same_font_data:
                        _log_console(
                            f"TTF 메타데이터 갱신: {assets_name} | {font.m_Name} | "
                            f"(PathID: {font_pathid} == {replacement_font})"
                        )
                    elif lang == "ko":
                        _log_console(
                            f"TTF 폰트 교체: {assets_name} | {font.m_Name} | (PathID: {font_pathid} -> {replacement_font})"
                        )
                    elif same_font_data:
                        _log_console(
                            f"TTF metadata updated: {assets_name} | {font.m_Name} | "
                            f"(PathID: {font_pathid} == {replacement_font})"
                        )
                    else:
                        _log_console(
                            f"TTF font replaced: {assets_name} | {font.m_Name} | (PathID: {font_pathid} -> {replacement_font})"
                        )
                    _log_debug(
                        f"[replace_ttf] file={fn_without_path} assets={assets_name} path_id={font_pathid} "
                        f"name={font.m_Name} target={replacement_font} "
                        f"old_size={len(current_ttf_data)} new_size={len(assets['ttf_data'])}"
                    )
                    if not same_font_data:
                        font.m_FontData = assets["ttf_data"]
                    _log_debug(
                        f"[replace_ttf] metadata path_id={font_pathid} "
                        f"font_names={metadata.get('font_names')} "
                        f"ascent={metadata.get('ascent')} "
                        f"descent={metadata.get('descent')} "
                        f"line_spacing={metadata.get('line_spacing')}"
                    )
                    _safe_save(obj, font)
                    satisfied_ttf_targets.add((str(assets_name), int(font_pathid)))
                    modified = True

        if obj.type.name == "MonoBehaviour" and replace_sdf:
            pathid = obj.path_id
            target_key = (assets_name, int(pathid))
            if target_sdf_targets and target_key not in target_sdf_targets:
                continue
            try:
                parse_dict = _safe_parse_as_dict(obj)
            except Exception as e:
                reason = f"PathID {obj.path_id} parse_as_dict 실패 [{type(e).__name__}]: {e!r}"
                sdf_parse_failure_reasons.append(reason)
                _log_debug(
                    f"[replace_sdf] file={fn_without_path} assets={assets_name} path_id={obj.path_id} "
                    f"action=parse_as_dict_failed error={type(e).__name__}: {e!r}"
                )
                if lang == "ko":
                    _log_console(f"  경고: {reason}")
                    debug_parse_log(
                        f"[replace_fonts] MonoBehaviour parse_as_dict 실패: {fn_without_path} | {reason}"
                    )
                else:
                    _log_console(
                        f"  Warning: PathID {obj.path_id} parse_as_dict failed [{type(e).__name__}]: {e!r}"
                    )
                    debug_parse_log(
                        f"[replace_fonts] MonoBehaviour parse_as_dict failed: {fn_without_path} | {reason}"
                    )
                continue
            unity_version_hint_raw = getattr(obj.assets_file, "unity_version", None)
            unity_version_hint = str(unity_version_hint_raw or unity_version or "")
            tmp_info = inspect_tmp_font_schema(
                parse_dict,
                unity_version=unity_version_hint or None,
            )
            if not tmp_info.get("is_tmp"):
                continue

            # KR: TMP_SpriteAsset은 SDF 폰트가 아니므로 교체 대상에서 제외합니다.
            #     spriteSheet, m_SpriteCharacterTable, spriteInfoList 중 하나라도 있으면 SpriteAsset입니다.
            # EN: Skip TMP_SpriteAsset — not an SDF font.
            #     Detected by presence of spriteSheet, m_SpriteCharacterTable, or spriteInfoList.
            if (
                parse_dict.get("spriteSheet") is not None
                or isinstance(parse_dict.get("m_SpriteCharacterTable"), list)
                or isinstance(parse_dict.get("spriteInfoList"), list)
            ):
                _log_debug(
                    f"[replace_sdf] file={fn_without_path} assets={assets_name} "
                    f"path_id={pathid} name={obj.peek_name()} "
                    f"action=skip_sprite_asset"
                )
                continue

            glyph_count = int(tmp_info.get("glyph_count", 0) or 0)
            atlas_file_id = int(tmp_info.get("atlas_file_id", 0) or 0)
            atlas_path_id = int(tmp_info.get("atlas_path_id", 0) or 0)

            # KR: 외부 참조 stub만 제외하고 실제 TMP 폰트만 처리합니다.
            # EN: Excludes only external reference stubs and processes actual TMP fonts.
            if atlas_file_id != 0 and atlas_path_id == 0:
                continue
            if glyph_count == 0:
                if atlas_file_id == 0 and atlas_path_id == 0:
                    continue

            objname = obj.peek_name()
            replacement_font = replacement_lookup.get(
                ("SDF", fn_without_path, assets_name, pathid)
            )
            if replacement_font is None:
                replacement_font = target_sdf_font_by_target.get(target_key)

            preview_target_meta = preview_target_lookup.get(
                (fn_without_path, assets_name, int(pathid))
            )
            if (
                replacement_font is None
                and preview_target_meta is not None
                and preview_export
            ):
                atlas_path_id_preview = int(tmp_info.get("atlas_path_id", 0) or 0)
                if atlas_path_id_preview:
                    target_swizzle_verdict: str | None = None
                    if ps5_swizzle:
                        target_swizzle_verdict, _ = _detect_target_texture_swizzle(
                            texture_object_lookup,
                            texture_swizzle_state_cache,
                            assets_name,
                            int(atlas_path_id_preview),
                        )
                    target_preview_image = _load_target_unswizzled_preview_image(
                        texture_object_lookup,
                        assets_name,
                        int(atlas_path_id_preview),
                        target_swizzle_verdict,
                        preview_rotate=PS5_SWIZZLE_ROTATE if ps5_swizzle else 0,
                    )
                    if isinstance(target_preview_image, Image.Image):
                        _save_swizzle_preview(
                            target_preview_image,
                            preview_enabled=preview_export,
                            preview_root=preview_root,
                            assets_file_name=fn_without_path,
                            assets_name=assets_name,
                            atlas_path_id=int(atlas_path_id_preview),
                            font_name=str(objname),
                            target_swizzled=bool(
                                target_swizzle_verdict == "likely_swizzled_input"
                            ),
                            lang=lang,
                        )
                        preview_sdf_data = normalize_sdf_data(parse_dict)
                        _save_glyph_crop_previews(
                            target_preview_image,
                            preview_enabled=preview_export,
                            preview_root=preview_root,
                            assets_file_name=fn_without_path,
                            assets_name=assets_name,
                            atlas_path_id=int(atlas_path_id_preview),
                            font_name=str(objname),
                            sdf_data=preview_sdf_data,
                            lang=lang,
                        )

            if replacement_font:
                replacement_meta = replacement_meta_lookup.get(
                    ("SDF", fn_without_path, assets_name, int(pathid)),
                    {},
                )
                replacement_process_swizzle = parse_bool_flag(
                    replacement_meta.get("process_swizzle")
                )
                replacement_swizzle_hint = parse_bool_flag(
                    replacement_meta.get("swizzle")
                )
                replacement_force_raster = parse_bool_flag(
                    replacement_meta.get("force_raster")
                )
                effective_force_raster = force_raster or replacement_force_raster
                _log_debug(
                    f"[replace_sdf] file={fn_without_path} assets={assets_name} path_id={pathid} "
                    f"font={objname} target={replacement_font} "
                    f"effective_force_raster={effective_force_raster} "
                    f"replacement_swizzle_hint={replacement_swizzle_hint} "
                    f"replacement_process_swizzle={replacement_process_swizzle}"
                )
                matched_sdf_targets += 1
                source_padding_hint = extract_tmp_atlas_padding(
                    parse_dict,
                    unity_version=unity_version_hint or None,
                )
                selected_padding_variant = (
                    _select_builtin_bulk_padding_variant(
                        replacement_font,
                        source_padding_hint,
                    )
                    if prefer_builtin_padding_variants
                    else None
                )
                replacement_asset_padding = select_replacement_asset_padding(
                    replacement_font,
                    source_padding_hint,
                    selected_padding_variant,
                )
                assets = load_font_assets(
                    replacement_font,
                    prefer_raster=effective_force_raster,
                    padding_variant=replacement_asset_padding,
                    charset_source=(
                        replacement_meta.get("Charset")
                        or replacement_meta.get("charset")
                        or charset_source
                    ),
                )
                if assets["sdf_data"] and assets["sdf_atlas"]:
                    if lang == "ko":
                        _log_console(
                            f"SDF 폰트 교체: {assets_name} | {objname} | (PathID: {pathid}) -> {replacement_font}"
                        )
                    else:
                        _log_console(
                            f"SDF font replaced: {assets_name} | {objname} | (PathID: {pathid}) -> {replacement_font}"
                        )
                    if selected_padding_variant is not None:
                        if lang == "ko":
                            _log_console(
                                f"  가장 가까운 내장 padding preset 선택: source {source_padding_hint:.2f} -> Padding_{selected_padding_variant}"
                            )
                        else:
                            _log_console(
                                f"  Selected nearest built-in padding preset: source {source_padding_hint:.2f} -> Padding_{selected_padding_variant}"
                            )
                    source_atlas = assets["sdf_atlas"]
                    source_swizzled = parse_bool_flag(assets.get("sdf_swizzle"))
                    asset_process_swizzle = parse_bool_flag(
                        assets.get("sdf_process_swizzle")
                    )
                    atlas_linear_for_alpha8 = source_atlas
                    if ps5_swizzle and source_swizzled:
                        try:
                            atlas_linear_for_alpha8 = apply_ps5_unswizzle_to_image(
                                source_atlas,
                                allow_axis_swap=True,
                                roughness_guard=True,
                            )
                        except Exception:
                            atlas_linear_for_alpha8 = source_atlas
                    # KR: 입력 JSON이 신형/구형이어도 내부 교체는 신형 TMP 스키마로 통일합니다.
                    # EN: Regardless of whether input JSON uses new or old format, internal replacement is unified to the new TMP schema.
                    replace_data = assets.get("sdf_data_normalized")
                    if not isinstance(replace_data, dict):
                        replace_data = normalize_sdf_data(assets["sdf_data"])
                    try:
                        replacement_render_mode = int(
                            replace_data.get("m_AtlasRenderMode", 4118) or 0
                        )
                    except Exception:
                        replacement_render_mode = 4118
                    if effective_force_raster:
                        replacement_render_mode &= ~0x1000
                    replacement_is_sdf = (replacement_render_mode & 0x1000) != 0
                    game_padding_for_material = 0.0

                    # KR: GameObject/Script/Material/Atlas 참조는 기존 PathID를 유지해야 런타임 연결이 깨지지 않습니다.
                    # EN: GameObject/Script/Material/Atlas references must keep existing PathIDs to avoid breaking runtime linkage.
                    m_GameObject_FileID = parse_dict["m_GameObject"]["m_FileID"]
                    m_GameObject_PathID = parse_dict["m_GameObject"]["m_PathID"]
                    m_Script_FileID = parse_dict["m_Script"]["m_FileID"]
                    m_Script_PathID = parse_dict["m_Script"]["m_PathID"]
                    has_source_font_ref = isinstance(
                        parse_dict.get("m_SourceFontFile"), dict
                    )
                    if has_source_font_ref:
                        m_SourceFontFile_FileID = int(
                            parse_dict["m_SourceFontFile"].get("m_FileID", 0) or 0
                        )
                        m_SourceFontFile_PathID = int(
                            parse_dict["m_SourceFontFile"].get("m_PathID", 0) or 0
                        )
                    else:
                        m_SourceFontFile_FileID = 0
                        m_SourceFontFile_PathID = 0

                    (
                        material_ref_key,
                        m_Material_FileID,
                        m_Material_PathID,
                    ) = _get_tmp_material_reference(
                        parse_dict
                    )

                    target_new_atlas_ref = _best_atlas_ref(
                        parse_dict,
                        prefer_new=True,
                    )
                    target_old_atlas_ref = (
                        cast(JsonDict, parse_dict.get("atlas"))
                        if isinstance(parse_dict.get("atlas"), dict)
                        else None
                    )
                    target_has_new_face = isinstance(parse_dict.get("m_FaceInfo"), dict)
                    target_has_new_glyphs = isinstance(
                        parse_dict.get("m_GlyphTable"), list
                    )
                    target_has_new_chars = isinstance(
                        parse_dict.get("m_CharacterTable"), list
                    )
                    target_has_old_face = isinstance(parse_dict.get("m_fontInfo"), dict)
                    target_has_old_glyphs = isinstance(
                        parse_dict.get("m_glyphInfoList"), list
                    )
                    target_creation_settings_key = _resolve_creation_settings_key(
                        parse_dict,
                        unity_version=unity_version_hint or None,
                    )
                    target_creation_settings = (
                        cast(JsonDict, parse_dict.get(target_creation_settings_key))
                        if target_creation_settings_key
                        and isinstance(
                            parse_dict.get(target_creation_settings_key), dict
                        )
                        else None
                    )

                    if target_new_atlas_ref is not None:
                        m_AtlasTextures_FileID, m_AtlasTextures_PathID = _atlas_ref_ids(
                            target_new_atlas_ref
                        )
                    elif target_old_atlas_ref is not None:
                        m_AtlasTextures_FileID, m_AtlasTextures_PathID = _atlas_ref_ids(
                            target_old_atlas_ref
                        )
                    else:
                        m_AtlasTextures_FileID = int(atlas_file_id)
                        m_AtlasTextures_PathID = int(atlas_path_id)

                    if target_has_new_face:
                        game_face_info = parse_dict.get("m_FaceInfo", {})
                        try:
                            game_padding_for_material = float(
                                parse_dict.get(
                                    "m_AtlasPadding",
                                    (
                                        target_creation_settings.get("padding", 0)
                                        if isinstance(target_creation_settings, dict)
                                        else 0
                                    ),
                                )
                            )
                        except Exception:
                            game_padding_for_material = 0.0

                        target_face_info = dict(replace_data["m_FaceInfo"])
                        if isinstance(game_face_info, dict):
                            if use_game_line_metrics:
                                metric_scale = 1.0
                            else:
                                metric_scale = _safe_metric_scale(
                                    game_face_info.get("m_PointSize", 0),
                                    target_face_info.get("m_PointSize", 0),
                                )
                            for metric_key in new_line_metric_keys:
                                if metric_key in game_face_info:
                                    metric_value = game_face_info[metric_key]
                                    if (
                                        metric_key in new_line_metric_scale_keys
                                        and metric_scale != 1.0
                                    ):
                                        try:
                                            metric_value = (
                                                float(metric_value) * metric_scale
                                            )
                                        except Exception:
                                            pass
                                    target_face_info[metric_key] = metric_value
                        ensure_int(
                            target_face_info,
                            ["m_PointSize", "m_AtlasWidth", "m_AtlasHeight"],
                        )
                        parse_dict["m_FaceInfo"] = target_face_info

                    replacement_glyph_table = copy.deepcopy(
                        replace_data.get("m_GlyphTable", [])
                        if isinstance(replace_data.get("m_GlyphTable", []), list)
                        else []
                    )
                    replacement_character_table = copy.deepcopy(
                        replace_data.get("m_CharacterTable", [])
                        if isinstance(replace_data.get("m_CharacterTable", []), list)
                        else []
                    )
                    nonzero_atlas_indexes = sorted(
                        {
                            int(glyph.get("m_AtlasIndex", 0) or 0)
                            for glyph in replacement_glyph_table
                            if isinstance(glyph, dict)
                            and int(glyph.get("m_AtlasIndex", 0) or 0) != 0
                        }
                    )
                    if nonzero_atlas_indexes:
                        raise ValueError(
                            "Multi-atlas replacement data is not supported; "
                            f"found m_AtlasIndex values {nonzero_atlas_indexes}."
                        )

                    if target_has_new_glyphs:
                        parse_dict["m_GlyphTable"] = replacement_glyph_table
                    if target_has_new_chars:
                        parse_dict["m_CharacterTable"] = replacement_character_table

                    if replacement_glyph_table:
                        replacement_glyph_indexes = [
                            int(g.get("m_Index", 0) or 0)
                            for g in replacement_glyph_table
                            if isinstance(g, dict)
                        ]
                        for glyph_index_key in _TMP_GLYPH_INDEX_LIST_KEYS:
                            if glyph_index_key in parse_dict:
                                parse_dict[glyph_index_key] = list(
                                    replacement_glyph_indexes
                                )
                    if "m_GlyphIndexListNewlyAdded" in parse_dict:
                        parse_dict["m_GlyphIndexListNewlyAdded"] = []

                    if "m_AtlasWidth" in parse_dict:
                        parse_dict["m_AtlasWidth"] = int(
                            replace_data.get(
                                "m_AtlasWidth", parse_dict.get("m_AtlasWidth", 0)
                            )
                            or 0
                        )
                    if "m_AtlasHeight" in parse_dict:
                        parse_dict["m_AtlasHeight"] = int(
                            replace_data.get(
                                "m_AtlasHeight", parse_dict.get("m_AtlasHeight", 0)
                            )
                            or 0
                        )
                    if "m_AtlasPadding" in parse_dict:
                        parse_dict["m_AtlasPadding"] = int(
                            replace_data.get(
                                "m_AtlasPadding", parse_dict.get("m_AtlasPadding", 0)
                            )
                            or 0
                        )
                    if "m_AtlasRenderMode" in parse_dict:
                        parse_dict["m_AtlasRenderMode"] = replacement_render_mode
                    if "m_UsedGlyphRects" in parse_dict:
                        parse_dict["m_UsedGlyphRects"] = replace_data.get(
                            "m_UsedGlyphRects", parse_dict.get("m_UsedGlyphRects", [])
                        )
                    if "m_FreeGlyphRects" in parse_dict:
                        parse_dict["m_FreeGlyphRects"] = replace_data.get(
                            "m_FreeGlyphRects", parse_dict.get("m_FreeGlyphRects", [])
                        )
                    if "m_FontWeightTable" in parse_dict:
                        parse_dict["m_FontWeightTable"] = replace_data.get(
                            "m_FontWeightTable", parse_dict.get("m_FontWeightTable", [])
                        )
                    for record_table_key in ("m_FontFeatureTable", "m_KerningTable"):
                        if record_table_key in parse_dict:
                            _sync_existing_record_table(
                                parse_dict.get(record_table_key),
                                replace_data.get(record_table_key),
                            )

                    if target_has_old_face or target_has_old_glyphs:
                        game_font_info = parse_dict.get("m_fontInfo", {})
                        if game_padding_for_material <= 0:
                            try:
                                game_padding_for_material = float(
                                    game_font_info.get(
                                        "Padding",
                                        (
                                            target_creation_settings.get("padding", 0)
                                            if isinstance(
                                                target_creation_settings, dict
                                            )
                                            else 0
                                        ),
                                    )
                                )
                            except Exception:
                                game_padding_for_material = 0.0

                        old_font_info = convert_face_info_new_to_old(
                            replace_data["m_FaceInfo"],
                            replace_data.get("m_AtlasPadding", 0),
                            replace_data.get("m_AtlasWidth", 0),
                            replace_data.get("m_AtlasHeight", 0),
                        )
                        if isinstance(game_font_info, dict):
                            if use_game_line_metrics:
                                metric_scale = 1.0
                            else:
                                metric_scale = _safe_metric_scale(
                                    game_font_info.get("PointSize", 0),
                                    old_font_info.get("PointSize", 0),
                                )
                            for metric_key in old_line_metric_keys:
                                if metric_key in game_font_info:
                                    metric_value = game_font_info[metric_key]
                                    if (
                                        metric_key in old_line_metric_scale_keys
                                        and metric_scale != 1.0
                                    ):
                                        try:
                                            metric_value = (
                                                float(metric_value) * metric_scale
                                            )
                                        except Exception:
                                            pass
                                    old_font_info[metric_key] = metric_value

                        replacement_atlas = assets.get("sdf_atlas")
                        atlas_height = int(
                            replace_data.get(
                                "m_AtlasHeight",
                                (
                                    replacement_atlas.height
                                    if replacement_atlas is not None
                                    else 0
                                ),
                            )
                        )
                        old_glyph_list = convert_glyphs_new_to_old(
                            replacement_glyph_table,
                            replacement_character_table,
                            atlas_height=atlas_height,
                        )
                        old_font_info["CharacterCount"] = len(old_glyph_list)
                        if target_has_old_face:
                            parse_dict["m_fontInfo"] = old_font_info
                        if target_has_old_glyphs:
                            parse_dict["m_glyphInfoList"] = old_glyph_list

                    if isinstance(target_creation_settings, dict):
                        atlas_width_for_cs = int(
                            parse_dict.get(
                                "m_AtlasWidth", replace_data.get("m_AtlasWidth", 0)
                            )
                            or 0
                        )
                        atlas_height_for_cs = int(
                            parse_dict.get(
                                "m_AtlasHeight", replace_data.get("m_AtlasHeight", 0)
                            )
                            or 0
                        )
                        padding_for_cs = int(
                            parse_dict.get(
                                "m_AtlasPadding", replace_data.get("m_AtlasPadding", 0)
                            )
                            or 0
                        )
                        if target_has_old_face and not use_game_line_metrics:
                            try:
                                padding_for_cs = int(
                                    parse_dict.get("m_fontInfo", {}).get(
                                        "Padding", padding_for_cs
                                    )
                                    or padding_for_cs
                                )
                            except Exception:
                                pass

                        point_size_for_cs = int(
                            replace_data.get("m_FaceInfo", {}).get("m_PointSize", 0)
                            or 0
                        )
                        if target_has_new_face:
                            point_size_for_cs = int(
                                parse_dict.get("m_FaceInfo", {}).get(
                                    "m_PointSize", point_size_for_cs
                                )
                                or point_size_for_cs
                            )
                        elif target_has_old_face:
                            point_size_for_cs = int(
                                parse_dict.get("m_fontInfo", {}).get(
                                    "PointSize", point_size_for_cs
                                )
                                or point_size_for_cs
                            )

                        _sync_creation_settings_payload(
                            target_creation_settings,
                            atlas_width=atlas_width_for_cs,
                            atlas_height=atlas_height_for_cs,
                            padding=padding_for_cs,
                            point_size=point_size_for_cs,
                        )

                    # KR: 신형/구형 필드가 공존하면 신형 face 기준으로 legacy face도 동기화합니다.
                    # EN: When both new and old fields coexist, synchronize the legacy face based on the new face info.
                    if target_has_new_face and target_has_old_face:
                        synced_old_face = convert_face_info_new_to_old(
                            parse_dict["m_FaceInfo"],
                            int(
                                parse_dict.get(
                                    "m_AtlasPadding",
                                    replace_data.get("m_AtlasPadding", 0),
                                )
                                or 0
                            ),
                            int(
                                parse_dict.get(
                                    "m_AtlasWidth", replace_data.get("m_AtlasWidth", 0)
                                )
                                or 0
                            ),
                            int(
                                parse_dict.get(
                                    "m_AtlasHeight",
                                    replace_data.get("m_AtlasHeight", 0),
                                )
                                or 0
                            ),
                            character_count=len(
                                parse_dict.get("m_glyphInfoList", [])
                                if isinstance(
                                    parse_dict.get("m_glyphInfoList"), list
                                )
                                else []
                            ),
                        )
                        parse_dict["m_fontInfo"] = synced_old_face

                    for dirty_key in _TMP_DIRTY_FLAG_KEYS:
                        if dirty_key in parse_dict:
                            parse_dict[dirty_key] = True

                    # KR: 포맷 분기 후 공통 참조를 원래 값으로 되돌립니다.
                    # EN: After format branching, restore common references to their original values.
                    parse_dict["m_GameObject"]["m_FileID"] = m_GameObject_FileID
                    parse_dict["m_GameObject"]["m_PathID"] = m_GameObject_PathID
                    parse_dict["m_Script"]["m_FileID"] = m_Script_FileID
                    parse_dict["m_Script"]["m_PathID"] = m_Script_PathID

                    if material_ref_key is not None:
                        parse_dict[material_ref_key]["m_FileID"] = m_Material_FileID
                        parse_dict[material_ref_key]["m_PathID"] = m_Material_PathID

                    if has_source_font_ref and isinstance(
                        parse_dict.get("m_SourceFontFile"), dict
                    ):
                        parse_dict["m_SourceFontFile"][
                            "m_FileID"
                        ] = m_SourceFontFile_FileID
                        parse_dict["m_SourceFontFile"][
                            "m_PathID"
                        ] = m_SourceFontFile_PathID

                    _sync_single_atlas_state(
                        parse_dict,
                        m_AtlasTextures_FileID,
                        m_AtlasTextures_PathID,
                        reference_template=target_new_atlas_ref,
                    )

                    atlas_metadata_width = int(source_atlas.width)
                    atlas_metadata_height = int(source_atlas.height)
                    texture_target_assets_name = _resolve_target_assets_name(
                        obj.assets_file,
                        assets_name,
                        int(m_AtlasTextures_FileID),
                    )
                    texture_target_file_key = _resolve_target_outer_file_key(
                        current_file_key,
                        obj.assets_file,
                        int(m_AtlasTextures_FileID),
                        texture_target_assets_name,
                        source_bundle_signature=source_bundle_signature,
                        asset_file_index=asset_file_index,
                    )
                    texture_key = ""
                    if (
                        int(m_AtlasTextures_PathID) != 0
                        and texture_target_assets_name
                        and texture_target_file_key
                    ):
                        texture_key = _make_assets_object_key(
                            texture_target_assets_name,
                            int(m_AtlasTextures_PathID),
                        )
                        texture_plan: JsonDict = {
                            "replacement_font": replacement_font,
                            "source_entry": f"{fn_without_path}|{assets_name}|{pathid}",
                            "font_name": str(objname),
                            "source_atlas": source_atlas,
                            "source_swizzled": bool(source_swizzled),
                            "replacement_swizzle_hint": bool(
                                replacement_swizzle_hint
                            ),
                            "replacement_process_swizzle": bool(
                                replacement_process_swizzle
                            ),
                            "asset_process_swizzle": bool(asset_process_swizzle),
                            "alpha8_linear_source": atlas_linear_for_alpha8,
                            "metadata_width": atlas_metadata_width,
                            "metadata_height": atlas_metadata_height,
                        }
                        if preview_export:
                            texture_plan["preview_sdf_data"] = replace_data
                        # KR: 같은 파일 대상도 PNG로 스필해 MonoBehaviour 패스 동안
                        #     여러 4096x4096 디코드 이미지를 동시에 보유하지 않습니다.
                        # EN: Spill same-file plans too, avoiding many decoded 4096x4096
                        #     images being retained throughout the MonoBehaviour pass.
                        texture_plan = _spill_deferred_texture_plan_to_disk(
                            texture_plan,
                            deferred_payload_dir,
                        )
                        _close_unique_images(source_atlas, atlas_linear_for_alpha8)
                        assets["sdf_atlas"] = None
                        source_atlas = None
                        atlas_linear_for_alpha8 = None
                        if texture_target_file_key == current_file_key:
                            (
                                stored_texture_plan,
                                inserted_texture_plan,
                            ) = _store_consistent_patch_value(
                                texture_patch_plans,
                                texture_key,
                                texture_plan,
                                patch_kind="texture",
                                target_file_key=current_file_key,
                                transaction=deferred_transaction,
                            )
                            if stored_texture_plan is not None:
                                # Incoming deferred payloads are owned by the
                                # caller and must survive a failed split retry.
                                # Only plans inserted by this invocation belong
                                # in the local cleanup bucket.
                                if inserted_texture_plan:
                                    _store_patch_value(
                                        owned_texture_patch_plans,
                                        texture_key,
                                        stored_texture_plan,
                                    )
                                required_local_texture_ids.add(
                                    id(stored_texture_plan)
                                )
                            else:
                                required_resolution_errors.append(
                                    f"conflicting texture target {texture_key}"
                                )
                                _cleanup_deferred_patch_bucket(
                                    {texture_key: texture_plan}
                                )
                        else:
                            if not _register_deferred_patch(
                                staged_texture_plans,
                                texture_target_file_key,
                                texture_key,
                                texture_plan,
                                pending_files=staged_pending_files,
                                patch_kind="texture",
                                transaction=deferred_transaction,
                            ):
                                required_resolution_errors.append(
                                    f"conflicting deferred texture target {texture_key}"
                                )
                                _cleanup_deferred_patch_bucket(
                                    {texture_key: texture_plan}
                                )
                    elif int(m_AtlasTextures_PathID) != 0:
                        resolution_error = (
                            f"atlas {m_AtlasTextures_FileID}:{m_AtlasTextures_PathID} "
                            "target could not be resolved"
                        )
                        required_resolution_errors.append(resolution_error)
                        _log_warning(
                            f"[replace_sdf] file={fn_without_path} assets={assets_name} "
                            f"path_id={pathid} atlas_ref={m_AtlasTextures_FileID}:{m_AtlasTextures_PathID} "
                            "could_not_resolve_texture_target=True"
                        )

                    atlas_fallback_payload: JsonDict = {
                        "w": atlas_metadata_width,
                        "h": atlas_metadata_height,
                        "gs": None,
                        "float_overrides": {},
                        "color_overrides": {},
                        "outline_ratio": outline_ratio,
                        "reset_keywords": False,
                        "prune_raster_material": False,
                        "preserve_gradient_floor": False,
                        "replacement_font": replacement_font,
                        "source_entry": f"{fn_without_path}|{assets_name}|{pathid}",
                    }
                    if texture_key and texture_target_file_key:
                        _store_consistent_patch_value(
                            material_replacements_by_atlas,
                            texture_key,
                            atlas_fallback_payload,
                            patch_kind="material_atlas",
                            target_file_key=current_file_key,
                            transaction=deferred_transaction,
                        )
                        if texture_target_file_key != current_file_key:
                            _register_deferred_patch(
                                staged_material_atlas_plans,
                                texture_target_file_key,
                                texture_key,
                                atlas_fallback_payload,
                                pending_files=staged_pending_files,
                                patch_kind="material_atlas",
                                transaction=deferred_transaction,
                            )
                        if collected_material_atlas_plans is not None:
                            _store_consistent_patch_value(
                                collected_material_atlas_plans,
                                texture_key,
                                atlas_fallback_payload,
                                patch_kind="material_atlas",
                                target_file_key="material_reconciliation",
                                transaction=None,
                            )
                    if m_Material_PathID != 0:
                        gradient_scale = None
                        apply_replacement_material = not use_game_mat
                        float_overrides: dict[str, float] = {}
                        color_overrides: dict[str, JsonDict] = {}
                        reset_keywords = False
                        prune_raster_material = False
                        preserve_gradient_floor = False
                        preserve_game_style = False
                        material_padding_ratio = 1.0
                        material_data = assets.get("sdf_materials")
                        if effective_force_raster and use_game_mat:
                            if lang == "ko":
                                _log_console(
                                    "  경고: Raster 폰트에 --use-game-material 사용 시 박스 아티팩트가 생길 수 있습니다."
                                )
                            else:
                                _log_console(
                                    "  Warning: using --use-game-material with Raster fonts may cause box artifacts."
                                )
                        try:
                            replacement_padding = float(
                                replace_data.get("m_AtlasPadding", 0)
                            )
                        except Exception:
                            replacement_padding = 0.0
                        if (
                            replacement_is_sdf
                            and game_padding_for_material > 0
                            and replacement_padding > 0
                            and game_padding_for_material > replacement_padding
                        ):
                            warn_key = (
                                str(assets_name),
                                str(objname),
                                int(pathid),
                            )
                            if warn_key not in replacement_padding_limit_warned:
                                replacement_padding_limit_warned.add(warn_key)
                                if lang == "ko":
                                    _log_console(
                                        "  경고: 원본 padding "
                                        f"{game_padding_for_material:.2f}가 교체 padding {replacement_padding:.2f}보다 큽니다. "
                                        "Material 보정을 적용하지만 외곽선/언더레이를 원본과 완전히 같게 복원하지 못할 수 있습니다."
                                    )
                                else:
                                    _log_console(
                                        "  Warning: source padding "
                                        f"{game_padding_for_material:.2f} exceeds replacement padding {replacement_padding:.2f}. "
                                        "Material correction is applied, but outline/underlay may not match the original exactly."
                                    )
                        if (
                            replacement_is_sdf
                            and material_scale_by_padding
                            and game_padding_for_material > 0
                            and replacement_padding > 0
                        ):
                            material_padding_ratio = (
                                game_padding_for_material / replacement_padding
                            )
                            if material_padding_ratio <= 0:
                                material_padding_ratio = 1.0
                        if material_data and apply_replacement_material:
                            preserve_game_style = (
                                replacement_is_sdf and (not effective_force_raster)
                            )
                            material_props = material_data.get("m_SavedProperties", {})
                            float_properties = material_props.get("m_Floats", [])
                            color_properties = material_props.get("m_Colors", [])
                            for prop in float_properties:
                                if not isinstance(prop, (list, tuple)) or len(prop) < 2:
                                    continue
                                key = str(prop[0])
                                if preserve_game_style and key in _MATERIAL_STYLE_FLOAT_KEYS:
                                    continue
                                try:
                                    value = float(prop[1])
                                except (TypeError, ValueError):
                                    continue
                                float_overrides[key] = value
                            for prop in color_properties:
                                if not isinstance(prop, (list, tuple)) or len(prop) < 2:
                                    continue
                                key = str(prop[0])
                                if preserve_game_style and key in _MATERIAL_STYLE_COLOR_KEYS:
                                    continue
                                color_value = _color_value_to_dict(
                                    prop[1],
                                    {"r": 0.0, "g": 0.0, "b": 0.0, "a": 0.0},
                                )
                                color_overrides[key] = color_value
                            if material_padding_ratio != 1.0:
                                for key in material_padding_scale_keys:
                                    if key in float_overrides:
                                        float_overrides[key] = float(
                                            float_overrides[key]
                                            * material_padding_ratio
                                        )
                            gradient_scale = float_overrides.get("_GradientScale")
                        # KR: 교체 material에 _GradientScale이 없으면 m_AtlasPadding+1로 자동 추론합니다.
                        # EN: If _GradientScale is missing from replacement material, auto-infer it as m_AtlasPadding+1.
                        if gradient_scale is None and replacement_is_sdf and replacement_padding > 0:
                            gradient_scale = float(replacement_padding + 1)
                        if apply_replacement_material and effective_force_raster:
                            # KR: Raster 모드에서는 SDF 계열 필드 0 덮기 대신 최소 필드만 남깁니다.
                            # EN: In Raster mode, keep only minimal fields instead of zeroing out SDF-related fields.
                            reset_keywords = True
                            prune_raster_material = True
                            gradient_scale = 1.0
                            if lang == "ko":
                                _log_console(
                                    "  Raster 모드 감지: Material 필드를 최소 구성으로 재구성합니다."
                                )
                            else:
                                _log_console(
                                    "  Raster mode detected: rebuilding Material to minimal raster-safe fields."
                                )
                        if (
                            apply_replacement_material
                            and replacement_is_sdf
                            and (not effective_force_raster)
                        ):
                            preserve_gradient_floor = True
                        if (
                            material_scale_by_padding
                            and apply_replacement_material
                            and material_padding_ratio != 1.0
                        ):
                            if lang == "ko":
                                _log_console(
                                    f"  Material padding 비율 보정 적용: {game_padding_for_material:.2f}/{replacement_padding:.2f} "
                                    f"(x{material_padding_ratio:.3f})"
                                )
                            else:
                                _log_console(
                                    f"  Applied material padding ratio: {game_padding_for_material:.2f}/{replacement_padding:.2f} "
                                    f"(x{material_padding_ratio:.3f})"
                                )
                        material_target_assets_name = _resolve_target_assets_name(
                            obj.assets_file,
                            assets_name,
                            int(m_Material_FileID),
                        )
                        material_target_file_key = _resolve_target_outer_file_key(
                            current_file_key,
                            obj.assets_file,
                            int(m_Material_FileID),
                            material_target_assets_name,
                            source_bundle_signature=source_bundle_signature,
                            asset_file_index=asset_file_index,
                        )
                        material_payload = {
                            "w": atlas_metadata_width,
                            "h": atlas_metadata_height,
                            "gs": gradient_scale,
                            "float_overrides": float_overrides,
                            "color_overrides": color_overrides,
                            "outline_ratio": outline_ratio,
                            "reset_keywords": reset_keywords,
                            "prune_raster_material": bool(prune_raster_material),
                            "preserve_game_style": bool(preserve_game_style),
                            "style_padding_scale_ratio": material_padding_ratio,
                            "preserve_gradient_floor": bool(
                                preserve_gradient_floor
                            ),
                            "replacement_padding": replacement_padding,
                            "replacement_font": replacement_font,
                            "source_entry": f"{fn_without_path}|{assets_name}|{pathid}",
                        }
                        if material_target_assets_name and material_target_file_key:
                            material_key_exact = _make_assets_object_key(
                                material_target_assets_name,
                                int(m_Material_PathID),
                            )
                            if material_target_file_key == current_file_key:
                                (
                                    stored_material_plan,
                                    _inserted_material_plan,
                                ) = _store_consistent_patch_value(
                                    material_replacements,
                                    material_key_exact,
                                    material_payload,
                                    patch_kind="material",
                                    target_file_key=current_file_key,
                                    transaction=deferred_transaction,
                                )
                                if stored_material_plan is not None:
                                    required_local_material_ids.add(
                                        id(stored_material_plan)
                                    )
                                else:
                                    required_resolution_errors.append(
                                        f"conflicting material target {material_key_exact}"
                                    )
                            else:
                                if not _register_deferred_patch(
                                    staged_material_plans,
                                    material_target_file_key,
                                    material_key_exact,
                                    material_payload,
                                    pending_files=staged_pending_files,
                                    patch_kind="material",
                                    transaction=deferred_transaction,
                                ):
                                    required_resolution_errors.append(
                                        "conflicting deferred material target "
                                        f"{material_key_exact}"
                                    )
                        elif material_target_file_key == current_file_key:
                            fallback_path_id = int(m_Material_PathID)
                            fallback_key = f"pathid|{fallback_path_id}"
                            existing_fallback = material_replacements_by_pathid.get(
                                fallback_path_id
                            )
                            fallback_consistent = not (
                                existing_fallback is not None
                                and _deferred_patch_fingerprint(
                                    "material", existing_fallback
                                )
                                != _deferred_patch_fingerprint(
                                    "material", material_payload
                                )
                            )
                            retained_fallback = (
                                existing_fallback
                                if existing_fallback is not None
                                else material_payload
                            )
                            if (
                                fallback_consistent
                                and (
                                    deferred_transaction is None
                                    or deferred_transaction.register_plan(
                                        "material",
                                        current_file_key,
                                        fallback_key,
                                        retained_fallback,
                                    )
                                )
                            ):
                                if (
                                    existing_fallback is not None
                                    and existing_fallback is not material_payload
                                ):
                                    _cleanup_superseded_patch_payload(
                                        material_payload,
                                        existing_fallback,
                                    )
                                material_replacements_by_pathid[fallback_path_id] = (
                                    retained_fallback
                                )
                                required_local_material_ids.add(id(retained_fallback))
                            else:
                                conflict = (
                                    f"conflicting material fallback target {fallback_key}"
                                )
                                required_resolution_errors.append(conflict)
                                if deferred_transaction is not None:
                                    deferred_transaction.fail(conflict)
                            _log_warning(
                                f"[replace_sdf] file={fn_without_path} assets={assets_name} path_id={pathid} "
                                f"material_ref={m_Material_FileID}:{m_Material_PathID} "
                                "could_not_resolve_material_assets_name=True; fallback_to_pathid_only=True"
                            )
                        else:
                            required_resolution_errors.append(
                                f"material {m_Material_FileID}:{m_Material_PathID} "
                                "target could not be resolved"
                            )
                            _log_warning(
                                f"[replace_sdf] file={fn_without_path} assets={assets_name} path_id={pathid} "
                                f"material_ref={m_Material_FileID}:{m_Material_PathID} "
                                "could_not_resolve_material_target=True"
                            )
                    obj.patch(parse_dict)
                    trailing = _pop_trailing_bytes(obj)
                    _append_trailing_bytes(obj, trailing)
                    patched_sdf_targets += 1
                    patched_sdf_target_keys.add(target_key)
                    modified = True
                else:
                    missing_parts: list[str] = []
                    if assets.get("sdf_data") is None:
                        missing_parts.append("json")
                    if assets.get("sdf_atlas") is None:
                        missing_parts.append("atlas")
                    if lang == "ko":
                        _log_console(
                            f"  경고: 교체 리소스 누락으로 SDF 적용 건너뜀: {replacement_font} "
                            f"(누락: {', '.join(missing_parts) if missing_parts else 'unknown'})"
                        )
                    else:
                        _log_console(
                            f"  Warning: skipping SDF patch due to missing replacement assets: {replacement_font} "
                            f"(missing: {', '.join(missing_parts) if missing_parts else 'unknown'})"
                        )

    phase_started_at = time.perf_counter()
    _emit_phase_callback(
        phase_callback,
        "patch_begin",
        file=fn_without_path,
        object_count=(
            len(getattr(env_file, "objects", {}))
            if hasattr(env_file, "objects")
            else None
        ),
    )
    for obj in env.objects:
        assets_name = obj.assets_file.name
        if obj.type.name == "Texture2D":
            replacement_key = _make_assets_object_key(assets_name, int(obj.path_id))
            texture_plan = _lookup_patch_value(texture_patch_plans, replacement_key)
            if isinstance(texture_plan, dict):
                parse_dict = _safe_parse_as_object(obj)
                typetree_size_mismatch = _detect_typetree_size_mismatch(
                    obj,
                    parse_dict,
                )
                if lang == "ko":
                    _log_console(
                        f"텍스처 교체: {obj.peek_name()} (PathID: {obj.path_id})"
                    )
                else:
                    _log_console(
                        f"Texture replaced: {obj.peek_name()} (PathID: {obj.path_id})"
                    )
                prepared_texture = _prepare_texture_replacement_for_target(
                    texture_plan,
                    assets_file_name=fn_without_path,
                    target_assets_name=assets_name,
                    target_path_id=int(obj.path_id),
                    texture_object_lookup=texture_object_lookup,
                    texture_swizzle_state_cache=texture_swizzle_state_cache,
                    ps5_swizzle=ps5_swizzle,
                    preview_export=preview_export,
                    preview_root=preview_root,
                    lang=lang,
                )
                if not isinstance(prepared_texture, dict):
                    continue
                replacement_image = prepared_texture.get("replacement_image")
                target_swizzled_state = prepared_texture.get(
                    "target_swizzled_state"
                )
                replacement_linear_source = prepared_texture.get(
                    "replacement_linear_source"
                )
                metadata_size = prepared_texture.get("metadata_size", (0, 0))
                if (
                    not isinstance(metadata_size, tuple)
                    or len(metadata_size) != 2
                ):
                    metadata_size = (0, 0)
                metadata_w, metadata_h = cast(tuple[int, int], metadata_size)
                applied_raw_alpha8 = False
                try:
                    texture_format = int(
                        getattr(parse_dict, "m_TextureFormat", -1) or -1
                    )
                except Exception:
                    texture_format = -1
                _log_debug(
                    f"[replace_texture] file={fn_without_path} assets={assets_name} path_id={obj.path_id} "
                    f"name={obj.peek_name()} texture_format={texture_format} metadata={metadata_w}x{metadata_h}"
                )
                if (
                    texture_format == 1
                    and isinstance(replacement_image, Image.Image)
                ):
                    try:
                        alpha_source = (
                            replacement_linear_source
                            if isinstance(replacement_linear_source, Image.Image)
                            else replacement_image
                        )
                        # KR: Alpha8은 반드시 bpe=1 경로로 인코딩해야 합니다.
                        #     RGBA 기준 swizzle 후 알파만 추출하면 바이트 순서가 깨질 수 있습니다.
                        # EN: Alpha8 must be encoded via the bpe=1 path.
                        #     Extracting only the alpha channel after RGBA-based swizzle can corrupt byte order.
                        alpha_raw, aw, ah, alpha_mode = _encode_alpha8_replacement_bytes(
                            alpha_source,
                            ps5_swizzle=ps5_swizzle,
                            target_swizzled_state=target_swizzled_state,
                        )
                        parse_dict.m_Width = int(metadata_w if metadata_w > 0 else aw)
                        parse_dict.m_Height = int(metadata_h if metadata_h > 0 else ah)
                        if hasattr(parse_dict, "m_CompleteImageSize"):
                            parse_dict.m_CompleteImageSize = int(len(alpha_raw))
                        parse_dict.image_data = alpha_raw
                        stream_data = getattr(parse_dict, "m_StreamData", None)
                        if stream_data is not None:
                            try:
                                stream_data.offset = 0
                                stream_data.size = 0
                                stream_data.path = ""
                            except Exception:
                                pass
                        applied_raw_alpha8 = True
                        _log_debug(
                            f"[replace_texture] file={fn_without_path} assets={assets_name} path_id={obj.path_id} "
                            f"action=alpha8_raw_injection target_swizzled={target_swizzled_state} "
                            f"mode={alpha_mode} raw_size={len(alpha_raw)} width={aw} height={ah}"
                        )
                        if lang == "ko":
                            if alpha_mode == "swizzled":
                                _log_console(
                                    "  Alpha8 raw 주입 적용: swizzled 바이트를 image_data에 직접 기록합니다."
                                )
                            elif alpha_mode == "linear_flipped":
                                _log_console(
                                    "  Alpha8 raw 주입 적용: linear 바이트(상하 반전 보정)를 image_data에 직접 기록합니다."
                                )
                            else:
                                _log_console(
                                    "  Alpha8 raw 주입 적용: 판정 불명(inconclusive) 상태로 image_data에 직접 기록합니다."
                                )
                        else:
                            if alpha_mode == "swizzled":
                                _log_console(
                                    "  Applied Alpha8 raw injection: writing swizzled bytes directly to image_data."
                                )
                            elif alpha_mode == "linear_flipped":
                                _log_console(
                                    "  Applied Alpha8 raw injection: writing linear bytes (with vertical-flip compensation) to image_data."
                                )
                            else:
                                _log_console(
                                    "  Applied Alpha8 raw injection: writing bytes directly to image_data (target state inconclusive)."
                                )
                    except Exception as raw_inject_error:
                        if lang == "ko":
                            _log_console(
                                f"  경고: Alpha8 raw 주입 실패, 일반 image 저장으로 폴백합니다. ({raw_inject_error})"
                            )
                        else:
                            _log_console(
                                f"  Warning: Alpha8 raw injection failed; falling back to image save. ({raw_inject_error})"
                            )
                if not applied_raw_alpha8:
                    parse_dict.image = replacement_image

                # KR: TypeTree 재직렬화 시 원본보다 작아지는 Texture2D (중국판 Unity 등)는
                # KR: 바이너리 패치를 사용하여 extra bytes를 보존합니다.
                # EN: For Texture2D that becomes smaller than original on TypeTree re-serialization (e.g. China Unity),
                # EN: use binary patching to preserve extra bytes.
                if _has_trailing_bytes(obj) or typetree_size_mismatch:
                    tex_w = int(getattr(parse_dict, "m_Width", 0) or 0)
                    tex_h = int(getattr(parse_dict, "m_Height", 0) or 0)
                    tex_image_data = getattr(parse_dict, "image_data", b"")
                    if not isinstance(tex_image_data, (bytes, bytearray)):
                        tex_image_data = bytes(tex_image_data)
                    if tex_image_data and tex_w > 0 and tex_h > 0:
                        if _binary_patch_texture2d(
                            obj,
                            image_data=tex_image_data,
                            width=tex_w,
                            height=tex_h,
                            lang=lang,
                        ):
                            if lang == "ko":
                                _log_console(
                                    "  바이너리 패치 적용 (TypeTree 외 extra bytes 보존)"
                                )
                            else:
                                _log_console(
                                    "  Applied binary patch (preserving extra bytes outside TypeTree)"
                                )
                        else:
                            _safe_save(obj, parse_dict)
                    else:
                        _safe_save(obj, parse_dict)
                else:
                    _safe_save(obj, parse_dict)
                consumed_texture_ids.add(id(texture_plan))
                modified = True
                parse_dict = None
                for owned_image in prepared_texture.get("_owned_images", []):
                    if isinstance(owned_image, Image.Image):
                        try:
                            owned_image.close()
                        except Exception:
                            pass
        if obj.type.name == "Material":
            parse_dict = None
            material_key = _make_assets_object_key(assets_name, int(obj.path_id))
            mat_info = _lookup_patch_value(material_replacements, material_key)
            if mat_info is None:
                fallback_path_id = int(obj.path_id)
                if fallback_path_id in material_replacements_by_pathid:
                    if material_object_count_by_pathid.get(fallback_path_id, 0) == 1:
                        mat_info = material_replacements_by_pathid[fallback_path_id]
                    elif fallback_path_id not in ambiguous_material_fallback_warned:
                        ambiguous_material_fallback_warned.add(fallback_path_id)
                        _log_warning(
                            f"[replace_material] file={fn_without_path} path_id={fallback_path_id} "
                            "fallback_pathid_only_match_ambiguous=True; skipped"
                        )
            if mat_info is None:
                if parse_dict is None:
                    parse_dict = _safe_parse_as_object(obj)
                atlas_key = _resolve_material_main_texture_key(
                    obj.assets_file,
                    assets_name,
                    parse_dict,
                )
                if atlas_key is not None:
                    mat_info = _lookup_patch_value(
                        material_replacements_by_atlas,
                        atlas_key,
                    )
            if mat_info is not None:
                if parse_dict is None:
                    parse_dict = _safe_parse_as_object(obj)
                if _apply_material_replacement_to_object(parse_dict, mat_info):
                    _safe_save(obj, parse_dict)
                    if id(mat_info) in (
                        incoming_material_ids | required_local_material_ids
                    ):
                        consumed_material_ids.add(id(mat_info))
                    modified = True

    # Atlas-keyed material plans are compatibility fallbacks. They are
    # best-effort and may legitimately have no Material in the texture file.
    handled_material_atlas_ids.update(incoming_material_atlas_ids)

    _emit_phase_callback(
        phase_callback,
        "patch_end",
        file=fn_without_path,
        elapsed_sec=(time.perf_counter() - phase_started_at),
        modified=bool(modified),
    )

    unresolved_texture_ids = (
        incoming_texture_ids | required_local_texture_ids
    ) - consumed_texture_ids
    unresolved_material_ids = (
        incoming_material_ids | required_local_material_ids
    ) - consumed_material_ids
    unsatisfied_ttf_targets = target_ttf_targets - satisfied_ttf_targets
    unsatisfied_sdf_targets = replacement_sdf_targets - patched_sdf_target_keys
    unstaged_transaction_required = bool(
        deferred_transaction is None
        and (staged_texture_plans or staged_material_plans)
    )
    unavailable_deferred_target_map = bool(
        (staged_texture_plans and not isinstance(deferred_texture_plans, dict))
        or (staged_material_plans and not isinstance(deferred_material_plans, dict))
    )
    deferred_plan_conflict = bool(
        deferred_transaction is not None and deferred_transaction.has_failures
    )
    save_blocked_by_deferred_patch = bool(
        required_resolution_errors
        or unresolved_texture_ids
        or unresolved_material_ids
        or unstaged_transaction_required
        or unavailable_deferred_target_map
        or deferred_plan_conflict
        or unsatisfied_ttf_targets
        or unsatisfied_sdf_targets
    )
    if save_blocked_by_deferred_patch:
        details = "; ".join(required_resolution_errors)
        _log_warning(
            f"[deferred_patch] refusing partial save for {fn_without_path}: "
            f"unmatched_textures={len(unresolved_texture_ids)} "
            f"unmatched_materials={len(unresolved_material_ids)}"
            f" transaction_missing={unstaged_transaction_required}"
            f" target_map_missing={unavailable_deferred_target_map}"
            f" plan_conflict={deferred_plan_conflict}"
            f" unmatched_ttf={len(unsatisfied_ttf_targets)}"
            f" unmatched_sdf={len(unsatisfied_sdf_targets)}"
            + (f" details={details}" if details else "")
        )
        if lang == "ko":
            _log_console(
                "  오류: 요청한 폰트 또는 외부 TMP 패치 일부를 확인할 수 없어 "
                "이 파일의 저장을 취소합니다."
            )
        else:
            _log_console(
                "  Error: cancelled this save because one or more requested font "
                "or external TMP patches could not be verified."
            )
        failure_reason = (
            f"{fn_without_path}: unmatched requested/deferred patch "
            f"(textures={len(unresolved_texture_ids)}, "
            f"materials={len(unresolved_material_ids)}, "
            f"ttf={len(unsatisfied_ttf_targets)}, "
            f"sdf={len(unsatisfied_sdf_targets)})"
        )
        if deferred_transaction is not None:
            deferred_transaction.fail(failure_reason)
        else:
            raise DeferredPatchAtomicityError(failure_reason)

    if modified and not save_blocked_by_deferred_patch:
        if lang == "ko":
            _log_console(f"'{fn_without_path}' 저장 중...")
        else:
            _log_console(f"Saving '{fn_without_path}'...")

        last_save_failure_reason: str | None = None

        def _save_env_file(
            packer: Any = None,
            save_path: str | None = None,
            use_save_to: bool = False,
        ) -> bytes | int:
            """KR: 지정 packer로 기본 파일 객체의 save/save_to를 호출합니다.
            save_path가 주어지면 save_to()로 파일에 직접 기록하여 메모리를 절약합니다.
            반환값은 bytes(legacy) 또는 저장된 파일 크기(int)입니다.

            EN: Invokes save/save_to on the base file object with the given packer.
            If save_path is given, writes directly to file via save_to() to save memory.
            Returns bytes (legacy) or the saved file size (int).
            """
            # KR: use_save_to=True 이고 save_to()가 존재하면 파일에 직접 저장합니다.
            # EN: If use_save_to=True and save_to() exists, save directly to file.
            save_to_fn = getattr(env_file, "save_to", None)
            if use_save_to and save_path and callable(save_to_fn):
                try:
                    supports_packer = (
                        "packer" in inspect.signature(save_to_fn).parameters
                    )
                except (TypeError, ValueError):
                    supports_packer = False
                if packer is None or not supports_packer:
                    return save_to_fn(save_path)
                return save_to_fn(save_path, packer=packer)

            # KR: 기존 bytes 반환 방식 폴백
            # EN: Fallback to legacy bytes-returning approach
            save_fn = getattr(env_file, "save", None)
            if not callable(save_fn):
                raise AttributeError(
                    "UnityPy environment file object has no callable save()."
                )
            typed_save = cast(Callable[..., bytes], save_fn)
            # KR: save() 시그니처를 기준으로 packer 지원 여부를 판별해 내부 TypeError를 가리지 않도록 합니다.
            # EN: Check packer support based on save() signature to avoid masking internal TypeErrors.
            try:
                supports_packer = "packer" in inspect.signature(typed_save).parameters
            except (TypeError, ValueError):
                supports_packer = False

            if packer is None or not supports_packer:
                return typed_save()
            return typed_save(packer=packer)

        def _validate_saved_file(saved_path: str) -> tuple[bool, str | None]:
            """KR: 저장 결과 Unity 파일이 다시 열리는지 검증합니다.
            EN: Validates that the saved Unity file can be re-opened.
            """
            signature = source_bundle_signature or getattr(env_file, "signature", None)
            saved_signature = _read_bundle_signature(saved_path, bundle_signatures)
            if signature in bundle_signatures and saved_signature != signature:
                reason = (
                    f"번들 시그니처 불일치 (기대: {signature}, 결과: {saved_signature or 'None'})"
                    if lang == "ko"
                    else f"bundle signature mismatch (expected: {signature}, got: {saved_signature or 'None'})"
                )
                if lang == "ko":
                    _log_console(f"  저장 검증 실패: {reason}")
                else:
                    _log_console(f"  Save validation failed: {reason}")
                return False, reason
            try:
                _emit_phase_callback(
                    phase_callback,
                    "validate_begin",
                    file=fn_without_path,
                    path=saved_path,
                )
                validation_started_at = time.perf_counter()
                validation_inner_names = _collect_validation_inner_names(env_file)
                if getattr(sys, "frozen", False):
                    cmd = [sys.executable, "--_validate-bundle", saved_path]
                else:
                    cmd = [
                        sys.executable,
                        os.path.abspath(__file__),
                        "--_validate-bundle",
                        saved_path,
                    ]
                for inner_name in validation_inner_names:
                    cmd.extend(["--_validate-inner-name", inner_name])
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=1800,
                )
                if proc.returncode == 0:
                    _emit_phase_callback(
                        phase_callback,
                        "validate_end",
                        file=fn_without_path,
                        path=saved_path,
                        elapsed_sec=(time.perf_counter() - validation_started_at),
                        ok=True,
                    )
                    return True, None
                detail = (proc.stderr or proc.stdout or "").strip()
                reason = (
                    f"worker exit={proc.returncode}: {detail}"
                    if detail
                    else f"worker exit={proc.returncode}"
                )
                if lang == "ko":
                    _log_console(f"  저장 검증 실패 [{reason}]")
                else:
                    _log_console(f"  Save validation failed [{reason}]")
                _emit_phase_callback(
                    phase_callback,
                    "validate_end",
                    file=fn_without_path,
                    path=saved_path,
                    elapsed_sec=(time.perf_counter() - validation_started_at),
                    ok=False,
                    reason=reason,
                )
                return False, reason
            except Exception as e:
                reason = (
                    f"검증 워커 실행 실패: {e!r}"
                    if lang == "ko"
                    else f"failed to run validation worker: {e!r}"
                )
                if lang == "ko":
                    _log_console(f"  저장 검증 워커 실행 실패: {e!r}")
                else:
                    _log_console(f"  Failed to run save validation worker: {e!r}")
                _emit_phase_callback(
                    phase_callback,
                    "validate_end",
                    file=fn_without_path,
                    path=saved_path,
                    ok=False,
                    reason=reason,
                )
                return False, reason

        def _try_save(packer_label: Any, log_label: str) -> bool:
            """KR: 단일 저장 전략을 시도하고 성공 여부를 반환합니다.
            EN: Attempts a single save strategy and returns whether it succeeded.
            """
            nonlocal save_success, last_save_failure_reason
            tmp_file = os.path.join(tmp_path, fn_without_path)
            has_save_to = callable(getattr(env_file, "save_to", None))
            try:
                _emit_phase_callback(
                    phase_callback,
                    "save_begin",
                    file=fn_without_path,
                    packer=packer_label,
                    method=log_label,
                )
                save_started_at = time.perf_counter()
                if not has_save_to:
                    raise RuntimeError(
                        "The loaded UnityPy file object does not expose save_to()."
                    )
                # KR: save_to() 실패 시 bytes 기반 save()로 되돌아가지 않습니다.
                #     현재 packer는 실패 처리하고 바깥 전략이 다음 packer를 시도합니다.
                # EN: Never fall back to bytes-returning save() after save_to() fails;
                #     fail this packer and let the outer strategy try the next one.
                _save_env_file(packer_label, save_path=tmp_file, use_save_to=True)
                gc.collect()
                is_valid, validation_reason = _validate_saved_file(tmp_file)
                if not is_valid:
                    last_save_failure_reason = (
                        validation_reason or "saved file validation failed"
                    )
                    try:
                        if os.path.exists(tmp_file):
                            os.remove(tmp_file)
                    except Exception:
                        pass
                    return False
                save_success = True
                _emit_phase_callback(
                    phase_callback,
                    "save_end",
                    file=fn_without_path,
                    packer=packer_label,
                    method=log_label,
                    elapsed_sec=(time.perf_counter() - save_started_at),
                    ok=True,
                    validated=True,
                )
                return True
            except Exception as e:
                last_save_failure_reason = (
                    f"method {log_label} [{type(e).__name__}]: {e!r}"
                )
                if lang == "ko":
                    _log_console(
                        f"  저장 방법 {log_label} 실패 [{type(e).__name__}]: {e!r}"
                    )
                else:
                    _log_console(
                        f"  Save method {log_label} failed [{type(e).__name__}]: {e!r}"
                    )
                if debug_parse_enabled():
                    tb_module.print_exc()
                try:
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)
                except Exception:
                    pass
                _emit_phase_callback(
                    phase_callback,
                    "save_end",
                    file=fn_without_path,
                    packer=packer_label,
                    method=log_label,
                    ok=False,
                    reason=last_save_failure_reason,
                )
                return False
            finally:
                gc.collect()

        dataflags = getattr(env_file, "dataflags", None)
        safe_none_packer = (int(dataflags), 0) if dataflags is not None else "none"
        legacy_none_packer = (
            ((int(dataflags) & ~0x3F), 0) if dataflags is not None else None
        )

        if prefer_original_compress:
            # KR: 옵션이 있으면 원본 압축 우선으로 저장합니다.
            # EN: If the option is set, save with original compression first.
            if not _try_save("original", "1"):
                if lang == "ko":
                    _log_console("  lz4 압축 모드로 재시도...")
                else:
                    _log_console("  Retrying with lz4 packer...")
                if not _try_save("lz4", "2"):
                    if lang == "ko":
                        _log_console("  비압축 계열 모드로 재시도...")
                    else:
                        _log_console("  Retrying with uncompressed-style packer...")
                    if (
                        not _try_save(safe_none_packer, "3")
                        and legacy_none_packer is not None
                    ):
                        if lang == "ko":
                            _log_console("  레거시 비트마스크 모드로 재시도...")
                        else:
                            _log_console("  Retrying with legacy bitmask packer...")
                        _try_save(legacy_none_packer, "4")
        else:
            # KR: 기본은 무압축 계열 우선으로 저장해 시간을 줄이고, 실패 시 압축 모드로 폴백합니다.
            # EN: By default, save with uncompressed-family first to reduce time; fall back to compressed mode on failure.
            if not _try_save(safe_none_packer, "1"):
                if legacy_none_packer is not None:
                    if lang == "ko":
                        _log_console("  레거시 비트마스크 무압축 모드로 재시도...")
                    else:
                        _log_console(
                            "  Retrying with legacy bitmask uncompressed packer..."
                        )
                    if _try_save(legacy_none_packer, "2"):
                        pass
                    else:
                        if lang == "ko":
                            _log_console("  원본 압축 모드로 재시도...")
                        else:
                            _log_console("  Retrying with original compression...")
                        if not _try_save("original", "3"):
                            if lang == "ko":
                                _log_console("  lz4 압축 모드로 재시도...")
                            else:
                                _log_console("  Retrying with lz4 packer...")
                            _try_save("lz4", "4")
                else:
                    if lang == "ko":
                        _log_console("  원본 압축 모드로 재시도...")
                    else:
                        _log_console("  Retrying with original compression...")
                    if not _try_save("original", "2"):
                        if lang == "ko":
                            _log_console("  lz4 압축 모드로 재시도...")
                        else:
                            _log_console("  Retrying with lz4 packer...")
                        _try_save("lz4", "3")

        close_unitypy_env(env)
        gc.collect()

        if save_success:
            saved_file_path = os.path.join(tmp_path, fn_without_path)
            if os.path.exists(saved_file_path):
                saved_size = os.path.getsize(saved_file_path)
                if deferred_transaction is not None:
                    deferred_transaction.backup(assets_file, replace_only=True)
                _atomic_replace_validated_file(saved_file_path, assets_file)
                _log_debug(
                    f"[save] file={fn_without_path} output={assets_file} temp={saved_file_path} bytes={saved_size}"
                )
                if lang == "ko":
                    _log_console(f"  저장 완료 (크기: {saved_size} bytes)")
                else:
                    _log_console(f"  Save complete (size: {saved_size} bytes)")
            else:
                _log_debug(
                    f"[save] file={fn_without_path} output={assets_file} temp={saved_file_path} missing_after_save=True"
                )
                if lang == "ko":
                    _log_console("  경고: 저장된 파일을 찾을 수 없습니다")
                else:
                    _log_console("  Warning: saved file was not found")
                last_save_failure_reason = "saved file was not found after save phase"
                save_success = False

        if not save_success:
            _log_debug(
                f"[save] file={fn_without_path} output={assets_file} failed=True reason={last_save_failure_reason}"
            )
            if lang == "ko":
                _log_console("  오류: 파일 저장에 실패했습니다.")
                if last_save_failure_reason:
                    _log_console(f"  실패 원인: {last_save_failure_reason}")
            else:
                _log_console("  Error: failed to save file.")
                if last_save_failure_reason:
                    _log_console(f"  Failure reason: {last_save_failure_reason}")
    elif replace_sdf and target_sdf_targets and not preview_export:
        if lang == "ko":
            _log_console(
                f"  경고: SDF 대상 {len(target_sdf_targets)}건 중 매칭 {matched_sdf_targets}건, 적용 {patched_sdf_targets}건"
            )
            if sdf_parse_failure_reasons:
                _log_console(f"  파싱 오류: {sdf_parse_failure_reasons[-1]}")
        else:
            _log_console(
                f"  Warning: SDF targets={len(target_sdf_targets)}, matched={matched_sdf_targets}, patched={patched_sdf_targets}"
            )
            if sdf_parse_failure_reasons:
                _log_console(f"  Parse error: {sdf_parse_failure_reasons[-1]}")

    if modified and save_success:
        _commit_staged_deferred_patches(
            staged_texture_plans,
            deferred_texture_plans,
            pending_files=pending_external_patch_files,
            patch_kind="texture",
            transaction=deferred_transaction,
        )
        _commit_staged_deferred_patches(
            staged_material_plans,
            deferred_material_plans,
            pending_files=pending_external_patch_files,
            patch_kind="material",
            transaction=deferred_transaction,
        )
        _commit_staged_deferred_patches(
            staged_material_atlas_plans,
            deferred_material_atlas_plans,
            pending_files=pending_external_patch_files,
            patch_kind="material_atlas",
            transaction=deferred_transaction,
        )
    else:
        for staged_map in (
            staged_texture_plans,
            staged_material_plans,
            staged_material_atlas_plans,
        ):
            for staged_bucket in staged_map.values():
                _cleanup_deferred_patch_bucket(staged_bucket)

    deferred_target_handled = (modified and save_success) or (
        not modified and not save_blocked_by_deferred_patch
    )
    if deferred_target_handled:
        _consume_deferred_patch_payloads(
            deferred_texture_plans,
            current_file_key,
            consumed_texture_ids & incoming_texture_ids,
        )
        _consume_deferred_patch_payloads(
            deferred_material_plans,
            current_file_key,
            consumed_material_ids & incoming_material_ids,
        )
        _consume_deferred_patch_payloads(
            deferred_material_atlas_plans,
            current_file_key,
            handled_material_atlas_ids,
        )

    # Only delete plans spilled by this call. Incoming deferred buckets belong
    # to the caller and must survive a failed one-shot save for split retries.
    _cleanup_deferred_patch_bucket(owned_texture_patch_plans)
    try:
        os.rmdir(deferred_payload_dir)
    except OSError:
        pass

    if operation_outcome is not None:
        requested_target_count = len(target_ttf_targets) + len(
            replacement_sdf_targets
        )
        satisfied_target_count = len(satisfied_ttf_targets) + len(
            patched_sdf_target_keys & replacement_sdf_targets
        )
        operation_outcome.update(
            {
                "requested_targets": requested_target_count,
                "satisfied_targets": satisfied_target_count,
                "modified": bool(modified),
                "save_success": bool(save_success),
                "already_satisfied": bool(
                    requested_target_count > 0
                    and satisfied_target_count == requested_target_count
                    and not modified
                    and not save_blocked_by_deferred_patch
                ),
            }
        )

    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    if not using_custom_temp_root and os.path.isdir(tmp_root):
        try:
            os.rmdir(tmp_root)
        except OSError:
            pass

    return save_success if modified else False


def create_batch_replacements(
    game_path: str,
    font_name: str,
    replace_ttf: bool = True,
    replace_sdf: bool = True,
    target_files: set[str] | None = None,
    exclude_exts: set[str] | None = None,
    scan_jobs: int = 1,
    lang: Language = "ko",
    ps5_swizzle: bool = False,
    scan_stall_seconds: float = DEFAULT_STALL_SECONDS,
) -> dict[str, JsonDict]:
    """KR: 게임 내 모든 폰트를 지정 폰트로 치환하는 배치 매핑을 생성합니다.
    target_files가 있으면 해당 파일만 대상으로 매핑을 생성합니다.
    exclude_exts가 있으면 해당 확장자는 스캔에서 제외합니다.

    EN: Creates a batch mapping to replace all fonts in the game with the specified font.
    If target_files is provided, only those files are included in the mapping.
    If exclude_exts is provided, those extensions are excluded from scanning.
    """
    fonts = scan_fonts(
        game_path,
        lang=lang,
        target_files=target_files,
        exclude_exts=exclude_exts,
        scan_jobs=scan_jobs,
        ps5_swizzle=ps5_swizzle,
        scan_ttf=replace_ttf,
        scan_sdf=replace_sdf,
        scan_stall_seconds=scan_stall_seconds,
    )
    replacements: dict[str, JsonDict] = {}

    if replace_ttf:
        for font in fonts["ttf"]:
            key = f"{font['file']}|TTF|{font['path_id']}"
            replacements[key] = {
                "Name": font["name"],
                "assets_name": font["assets_name"],
                "Path_ID": font["path_id"],
                "Type": "TTF",
                "File": font["file"],
                "Replace_to": font_name,
            }

    if replace_sdf:
        for font in fonts["sdf"]:
            key = f"{font['file']}|SDF|{font['path_id']}"
            if ps5_swizzle:
                swizzle_flag = (
                    "True" if parse_bool_flag(font.get("swizzle")) else "False"
                )
                process_swizzle_flag = (
                    "True" if parse_bool_flag(font.get("process_swizzle")) else "False"
                )
                entry: JsonDict = {
                    "File": font["file"],
                    "assets_name": font["assets_name"],
                    "Path_ID": font["path_id"],
                    "Type": "SDF",
                    "Name": font["name"],
                    "force_raster": "False",
                    "swizzle": swizzle_flag,
                    "process_swizzle": process_swizzle_flag,
                    "Replace_to": font_name,
                }
            else:
                entry = {
                    "File": font["file"],
                    "assets_name": font["assets_name"],
                    "Path_ID": font["path_id"],
                    "Type": "SDF",
                    "Name": font["name"],
                    "force_raster": "False",
                    "Replace_to": font_name,
                }
            replacements[key] = entry

    return replacements


def create_preview_export_targets(
    game_path: str,
    target_files: set[str] | None = None,
    exclude_exts: set[str] | None = None,
    scan_jobs: int = 1,
    lang: Language = "ko",
    ps5_swizzle: bool = False,
    scan_stall_seconds: float = DEFAULT_STALL_SECONDS,
) -> dict[str, JsonDict]:
    """KR: preview-export 전용 SDF 대상 매핑(Replace_to 비어 있음)을 생성합니다.
    scan_jobs/target_files/exclude_exts 조건을 그대로 반영합니다.

    EN: Creates an SDF target mapping for preview-export (Replace_to left empty).
    Honors scan_jobs/target_files/exclude_exts conditions as-is.
    """
    fonts = scan_fonts(
        game_path,
        lang=lang,
        target_files=target_files,
        exclude_exts=exclude_exts,
        scan_jobs=scan_jobs,
        ps5_swizzle=ps5_swizzle,
        scan_stall_seconds=scan_stall_seconds,
    )
    targets: dict[str, JsonDict] = {}
    for font in fonts["sdf"]:
        key = f"{font['file']}|PREVIEW|{font['path_id']}"
        entry: JsonDict = {
            "File": font["file"],
            "assets_name": font["assets_name"],
            "Path_ID": font["path_id"],
            "Type": "SDF",
            "Name": font["name"],
            "force_raster": "False",
            "Replace_to": "",
        }
        if ps5_swizzle:
            entry["swizzle"] = (
                "True" if parse_bool_flag(font.get("swizzle")) else "False"
            )
            entry["process_swizzle"] = (
                "True" if parse_bool_flag(font.get("process_swizzle")) else "False"
            )
        targets[key] = entry
    return targets


def exit_with_error(
    message: str,
    lang: Language = "ko",
    *,
    pause: bool | None = None,
) -> NoReturn:
    """KR: 로컬라이즈된 오류 메시지를 출력하고 종료합니다.
    EN: Prints a localized error message and exits.
    """
    if lang == "ko":
        _log_console(f"오류: {message}")
    else:
        _log_console(f"Error: {message}")
    if pause is None:
        pause = _should_pause_before_exit(interactive_session=False)
    if pause:
        _pause_before_exit(lang=lang, interactive_session=False)
    sys.exit(1)


def exit_with_error_en(message: str) -> NoReturn:
    """KR: 영문 오류 메시지를 출력하고 종료합니다.
    EN: Prints an English error message and exits.
    """
    exit_with_error(message, lang="en")


@cleanup_unitypy_environments
def run_validation_worker(
    bundle_path: str,
    lang: Language = "ko",
    inner_names: list[str] | None = None,
) -> int:
    """KR: 저장 검증 전용 워커입니다. 가능한 경우 경량 structural 검증을 수행합니다.
    EN: Dedicated save-validation worker. Performs lightweight structural validation when possible.
    """
    try:
        if not os.path.exists(bundle_path):
            if lang == "ko":
                _log_console("[validate] 검증 실패: 저장 파일이 존재하지 않습니다.")
            else:
                _log_console("[validate] Validation failed: saved file does not exist.")
            return 2

        signature = _read_bundle_signature(bundle_path, BUNDLE_SIGNATURES)
        if signature == "UnityFS":
            ok, reason = _structural_validate_unityfs_bundle(
                bundle_path,
                inner_names=inner_names,
            )
            if ok:
                return 0
            if lang == "ko":
                _log_console(f"[validate] structural 검증 실패: {reason}")
            else:
                _log_console(f"[validate] Structural validation failed: {reason}")
            return 2

        env = load_unitypy(bundle_path)
        files = getattr(env, "files", None)
        if not isinstance(files, dict) or len(files) == 0:
            if lang == "ko":
                _log_console(
                    "[validate] 검증 실패: UnityPy.load 결과에 파일이 없습니다."
                )
            else:
                _log_console(
                    "[validate] Validation failed: UnityPy.load returned no files."
                )
            return 2
        if not getattr(env, "objects", None):
            if lang == "ko":
                _log_console("[validate] 검증 실패: 로드된 오브젝트가 없습니다.")
            else:
                _log_console(
                    "[validate] Validation failed: loaded object list is empty."
                )
            return 2
        return 0
    except Exception as e:
        if lang == "ko":
            _log_console(f"[validate] 검증 실패: {e!r}")
        else:
            _log_console(f"[validate] Validation failed: {e!r}")
        if debug_parse_enabled():
            tb_module.print_exc()
        return 2


def run_persistent_scan_worker(
    game_path: str,
    lang: Language = "ko",
    detect_ps5_swizzle: bool = False,
    scan_ttf: bool = True,
    scan_sdf: bool = True,
) -> int:
    """KR: 여러 파일을 순차 처리하는 JSON-lines 영구 스캔 워커입니다.
    EN: JSON-lines persistent scan worker that handles multiple files.
    """
    try:
        game_path, data_path = resolve_game_path(game_path, lang=lang)
        generator: TypeTreeGenerator | None = None
        if scan_sdf:
            unity_version = get_unity_version(game_path, lang=lang)
            compile_method = get_compile_method(data_path)
            generator = _create_generator(
                unity_version,
                game_path,
                data_path,
                compile_method,
                lang=lang,
            )
    except Exception as error:
        write_protocol_message(
            sys.stdout,
            {
                "type": "fatal",
                "error": repr(error),
            },
        )
        return 2

    write_protocol_message(
        sys.stdout,
        {
            "type": "ready",
            "pid": os.getpid(),
        },
    )
    for raw_line in sys.stdin:
        message = decode_protocol_message(raw_line)
        if message is None:
            continue
        message_type = message.get("type")
        if message_type == "shutdown":
            return 0
        if message_type != "scan":
            continue

        job_id = message.get("job_id")
        assets_file = str(message.get("path", "")).strip()
        if not assets_file:
            write_protocol_message(
                sys.stdout,
                {
                    "type": "result",
                    "job_id": job_id,
                    "payload": {
                        "ttf": [],
                        "sdf": [],
                        "error": "scan worker request has an empty asset path",
                    },
                },
            )
            continue

        def _worker_phase_callback(phase: str, _payload: JsonDict) -> None:
            write_protocol_message(
                sys.stdout,
                {
                    "type": "activity",
                    "job_id": job_id,
                    "phase": phase,
                },
            )

        try:
            scanned, load_error = _scan_fonts_in_asset_file(
                assets_file,
                generator,
                lang=lang,
                detect_ps5_swizzle=detect_ps5_swizzle,
                scan_ttf=scan_ttf,
                scan_sdf=scan_sdf,
                phase_callback=_worker_phase_callback,
            )
            payload: JsonDict = {
                "ttf": scanned.get("ttf", []),
                "sdf": scanned.get("sdf", []),
                "error": load_error,
            }
        except Exception as error:
            if debug_parse_enabled():
                tb_module.print_exc()
            payload = {
                "ttf": [],
                "sdf": [],
                "error": f"scan worker caught exception: {error!r}",
            }

        write_protocol_message(
            sys.stdout,
            {
                "type": "result",
                "job_id": job_id,
                "payload": payload,
            },
        )
    return 0


def run_scan_file_worker(
    game_path: str,
    assets_file: str,
    output_path: str,
    lang: Language = "ko",
    detect_ps5_swizzle: bool = False,
    scan_ttf: bool = True,
    scan_sdf: bool = True,
) -> int:
    """KR: 단일 파일 파싱 워커입니다. 결과를 JSON 파일로 저장합니다.
    EN: Single-file parsing worker. Saves results to a JSON file.
    """
    try:
        game_path, data_path = resolve_game_path(game_path, lang=lang)
        generator: TypeTreeGenerator | None = None
        if scan_sdf:
            unity_version = get_unity_version(game_path, lang=lang)
            compile_method = get_compile_method(data_path)
            generator = _create_generator(
                unity_version, game_path, data_path, compile_method, lang=lang
            )
        scanned, load_error = _scan_fonts_in_asset_file(
            assets_file,
            generator,
            lang=lang,
            detect_ps5_swizzle=detect_ps5_swizzle,
            scan_ttf=scan_ttf,
            scan_sdf=scan_sdf,
        )
        payload: JsonDict = {
            "ttf": scanned.get("ttf", []),
            "sdf": scanned.get("sdf", []),
            "error": load_error,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        return 0
    except Exception as e:
        if lang == "ko":
            _log_console(f"[scan_worker] 실패: {e!r}")
        else:
            _log_console(f"[scan_worker] failed: {e!r}")
        if debug_parse_enabled():
            tb_module.print_exc()
        return 2


@_rollback_deferred_transaction_on_exit
def main_cli(lang: Language = "ko") -> None:
    """KR: 언어별 공통 CLI 진입점입니다.
    EN: Common CLI entry point per language.
    """
    is_ko = lang == "ko"

    if is_ko:
        description = "Unity 게임의 폰트를 한글 폰트로 교체합니다."
        epilog = """
예시:
  %(prog)s --gamepath "C:/path/to/game" --parse
  %(prog)s --gamepath "C:/path/to/game" --preview-export
  %(prog)s --gamepath "C:/path/to/game" --mulmaru
  %(prog)s --gamepath "C:/path/to/game" --nanumgothic --sdfonly
  %(prog)s --gamepath "C:/path/to/game" --font "D:/Fonts/MyFont.ttf"
  %(prog)s --gamepath "C:/path/to/game" --list font_map.json
        """
        gamepath_help = "게임의 루트 경로 (예: C:/path/to/game)"
        parse_help = "폰트 정보를 JSON으로 출력"
        mulmaru_help = "모든 폰트를 Mulmaru로 일괄 교체"
        nanum_help = "모든 폰트를 NanumGothic으로 일괄 교체"
        font_help = "지정한 폰트 이름/TTF/OTF로 모든 폰트를 일괄 교체"
        sdf_help = "SDF 폰트만 교체"
        ttf_help = "TTF 폰트만 교체"
        list_help = "JSON 파일을 읽어서 폰트 교체"
        target_file_help = "지정한 파일명만 교체 대상에 포함 (여러 번 사용 가능)"
        exclude_ext_help = (
            "스캔 제외 확장자 목록 (콤마 구분, 예: \"resS,.resource\")"
        )
        charset_help = "TTF/OTF에서 SDF 자동 생성 시 사용할 글자셋 파일 또는 직접 문자열 (기본: CharList_3911.txt)"
        game_mat_help = "SDF 교체 시 게임 원본 Material 파라미터를 보정 없이 그대로 유지 (기본: 원본 스타일 유지 + atlas/padding 자동 보정)"
        force_raster_help = "SDF 교체 시 교체 폰트를 Raster 모드로 강제 (렌더 모드/Material 효과값 Raster 기준 적용)"
        game_line_metrics_help = "SDF 교체 시 게임 원본 줄 간격 메트릭 사용 (기본: 교체 폰트 메트릭 보정 적용)"
        outline_ratio_help = (
            "SDF 외곽선 비율 배율 (기본: 1.0, _OutlineWidth/_OutlineSoftness에 적용)"
        )
        original_compress_help = (
            "저장 시 원본 압축 모드를 우선 사용 (기본: 무압축 계열 우선)"
        )
        temp_dir_help = "임시 저장 폴더 루트 경로 (가능하면 빠른 SSD/NVMe 권장)"
        output_only_help = (
            "원본 파일은 유지하고, 수정된 파일만 지정 폴더에 원본 상대 경로로 저장"
        )
        preview_help = "모든 SDF 폰트 Atlas/Glyph crop 미리보기를 preview 폴더에 저장 (--ps5-swizzle와 함께면 unswizzle 기준)"
        scan_jobs_help = "폰트 스캔 병렬 워커 수 (기본: 1, parse/일괄교체 스캔에 적용, 별칭: --max-workers)"
        scan_stall_help = (
            "CPU/I/O/진행 신호가 모두 멈춘 워커의 정지 판정 시간(초, 기본: 300, "
            "0이면 비활성화; 파일 총 처리시간 제한이 아님)"
        )
        split_save_force_help = (
            "대형 SDF 다건 교체에서 one-shot을 건너뛰고 SDF 1개씩 강제 분할 저장"
        )
        oneshot_save_force_help = (
            "대형 SDF 다건 교체에서도 분할 저장 폴백 없이 one-shot 저장만 시도"
        )
        ps5_swizzle_help = "PS5 swizzle 자동 판별/변환 모드 (mask_x=0x385F0, mask_y=0x07A0F, rotate=90 보정)"
        verbose_help = "콘솔 로그는 유지하고, 상세 DEBUG 로그(파일/폰트/경로/버전)를 verbose.txt에 저장"
    else:
        description = "Replace Unity game fonts with Korean fonts."
        epilog = """
Examples:
  %(prog)s --gamepath "C:/path/to/game" --parse
  %(prog)s --gamepath "C:/path/to/game" --preview-export
  %(prog)s --gamepath "C:/path/to/game" --mulmaru
  %(prog)s --gamepath "C:/path/to/game" --nanumgothic --sdfonly
  %(prog)s --gamepath "C:/path/to/game" --font "D:/Fonts/MyFont.ttf"
  %(prog)s --gamepath "C:/path/to/game" --list font_map.json
        """
        gamepath_help = "Game root path (e.g. C:/path/to/game)"
        parse_help = "Export font info to JSON"
        mulmaru_help = "Replace all fonts with Mulmaru"
        nanum_help = "Replace all fonts with NanumGothic"
        font_help = "Bulk replace all fonts with this font name/TTF/OTF"
        sdf_help = "Replace SDF fonts only"
        ttf_help = "Replace TTF fonts only"
        list_help = "Replace fonts using a JSON file"
        target_file_help = (
            "Limit replacement targets to specific file name(s) (repeatable)"
        )
        exclude_ext_help = (
            "Additional scan-excluded extensions (comma-separated, e.g. \"resS,.resource\")"
        )
        charset_help = "Charset file or literal characters for TTF/OTF-to-SDF auto-generation (default: CharList_3911.txt)"
        game_mat_help = "Keep original in-game Material parameters without correction for SDF replacement (default: preserve original style with automatic atlas/padding correction)"
        force_raster_help = "Force replacement fonts into Raster mode for SDF replacement (render mode/material effects follow Raster behavior)"
        game_line_metrics_help = "Use original in-game line metrics for SDF replacement (default: adjusted replacement font metrics)"
        outline_ratio_help = (
            "SDF outline ratio multiplier (default: 1.0, applied to _OutlineWidth/_OutlineSoftness)"
        )
        original_compress_help = "Prefer original compression mode on save (default: uncompressed-family first)"
        temp_dir_help = "Root path for temporary save files (fast SSD/NVMe recommended)"
        output_only_help = "Keep originals untouched and write modified files only to this folder (preserve relative paths)"
        preview_help = "Export preview PNGs (Atlas + glyph crops) for all SDF fonts into preview folder (unswizzled when used with --ps5-swizzle)"
        scan_jobs_help = "Number of parallel scan workers (default: 1, used for parse/bulk scan paths, alias: --max-workers)"
        scan_stall_help = (
            "Worker inactivity threshold in seconds based on CPU/I/O/progress "
            "(default: 300, 0 disables; not a total per-file runtime limit)"
        )
        split_save_force_help = "Skip one-shot and force one-by-one SDF split save for large multi-SDF replacements"
        oneshot_save_force_help = "Force one-shot save even for large multi-SDF targets (disable split-save fallback)"
        ps5_swizzle_help = "Enable PS5 swizzle detect/transform mode (mask_x=0x385F0, mask_y=0x07A0F, rotate=90 compensation)"
        verbose_help = "Keep concise console logs and save detailed DEBUG logs (file/font/path/version) to verbose.txt"

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument("--gamepath", type=str, help=gamepath_help)
    parser.add_argument("--parse", action="store_true", help=parse_help)
    parser.add_argument("--mulmaru", action="store_true", help=mulmaru_help)
    parser.add_argument("--nanumgothic", action="store_true", help=nanum_help)
    parser.add_argument("--font", type=str, metavar="FONT", help=font_help)
    parser.add_argument("--sdfonly", action="store_true", help=sdf_help)
    parser.add_argument("--ttfonly", action="store_true", help=ttf_help)
    parser.add_argument("--list", type=str, metavar="JSON_FILE", help=list_help)
    parser.add_argument(
        "--target-file", action="append", metavar="FILE_NAME", help=target_file_help
    )
    parser.add_argument(
        "--exclude-ext", action="append", metavar="EXTS", help=exclude_ext_help
    )
    parser.add_argument("--charset", type=str, metavar="PATH_OR_TEXT", help=charset_help)
    parser.add_argument("--use-game-material", action="store_true", help=game_mat_help)
    parser.add_argument("--force-raster", action="store_true", help=force_raster_help)
    parser.add_argument("--use-game-mat", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--use-game-line-metrics", action="store_true", help=game_line_metrics_help
    )
    parser.add_argument(
        "--outline-ratio",
        type=float,
        default=1.0,
        metavar="RATIO",
        help=outline_ratio_help,
    )
    parser.add_argument(
        "--use-game-line-matrics", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--original-compress", action="store_true", help=original_compress_help
    )
    parser.add_argument("--temp-dir", type=str, metavar="PATH", help=temp_dir_help)
    parser.add_argument(
        "--output-only", type=str, metavar="PATH", help=output_only_help
    )
    parser.add_argument("--preview-export", action="store_true", help=preview_help)
    parser.add_argument("--preview", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--scan-jobs",
        "--max-workers",
        dest="scan_jobs",
        type=int,
        default=1,
        metavar="N",
        help=scan_jobs_help,
    )
    parser.add_argument(
        "--scan-stall-seconds",
        type=float,
        default=DEFAULT_STALL_SECONDS,
        metavar="SECONDS",
        help=scan_stall_help,
    )
    parser.add_argument(
        "--split-save-force", action="store_true", help=split_save_force_help
    )
    parser.add_argument(
        "--oneshot-save-force", action="store_true", help=oneshot_save_force_help
    )
    parser.add_argument("--ps5-swizzle", action="store_true", help=ps5_swizzle_help)
    parser.add_argument("--verbose", action="store_true", help=verbose_help)
    parser.add_argument(
        "--_validate-bundle", type=str, metavar="BUNDLE_PATH", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--_validate-inner-name",
        action="append",
        metavar="INNER_NAME",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_scan-file-worker",
        type=str,
        metavar="ASSET_FILE_PATH",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_scan-worker-server",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_scan-worker-lang",
        choices=("ko", "en"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_scan-file-worker-output",
        type=str,
        metavar="OUTPUT_JSON_PATH",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_scan-ttf-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_scan-sdf-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()
    if isinstance(args.gamepath, str):
        args.gamepath = strip_wrapping_quotes_repeated(args.gamepath)
    if isinstance(args.list, str):
        args.list = strip_wrapping_quotes_repeated(args.list)
    if isinstance(args.charset, str):
        args.charset = strip_wrapping_quotes_repeated(args.charset)
    if isinstance(args.output_only, str):
        args.output_only = strip_wrapping_quotes_repeated(args.output_only)
    if isinstance(args.font, str):
        args.font = strip_wrapping_quotes_repeated(args.font)
    if isinstance(getattr(args, "exclude_ext", None), list):
        args.exclude_ext = [
            strip_wrapping_quotes_repeated(str(item))
            for item in args.exclude_ext
            if str(item).strip()
        ]

    verbose_path: str | None = None
    if args.verbose:
        verbose_path = os.path.join(get_script_dir(), VERBOSE_LOG_FILENAME)
    _configure_logging(
        console_level=logging.INFO,
        verbose_log_path=verbose_path,
    )
    py_bits = struct.calcsize("P") * 8
    _log_console(f"Python {sys.version} ({py_bits}-bit)")

    if verbose_path:
        if is_ko:
            _log_info(f"[verbose] 상세 로그를 '{verbose_path}'에 저장합니다.")
        else:
            _log_info(f"[verbose] Writing detailed logs to '{verbose_path}'.")
    _log_debug(
        f"[runtime] cwd={os.getcwd()} script_dir={get_script_dir()} args={vars(args)}"
    )

    # KR: 이전 옵션(--use-game-mat) 호환을 위해 새 옵션에 병합합니다.
    # EN: Merge the legacy option (--use-game-mat) into the new option for backward compatibility.
    args.use_game_material = bool(
        getattr(args, "use_game_material", False)
        or getattr(args, "use_game_mat", False)
    )
    # KR: 오타/레거시 옵션(--use-game-line-matrics)도 동일 동작으로 병합합니다.
    # EN: Also merge the typo/legacy option (--use-game-line-matrics) with the same behavior.
    args.use_game_line_metrics = bool(
        getattr(args, "use_game_line_metrics", False)
        or getattr(args, "use_game_line_matrics", False)
    )
    # KR: 레거시 옵션(--preview)도 새 옵션(--preview-export)으로 병합합니다.
    # EN: Also merge the legacy option (--preview) into the new option (--preview-export).
    args.preview_export = bool(
        getattr(args, "preview_export", False) or getattr(args, "preview", False)
    )
    explicit_primary_modes = _selected_primary_modes(args)
    if len(explicit_primary_modes) > 1:
        joined = ", ".join(explicit_primary_modes)
        if is_ko:
            exit_with_error(
                f"작업 모드 인자는 하나만 사용할 수 있습니다: {joined}",
                lang=lang,
            )
        else:
            exit_with_error(
                f"Only one primary mode may be selected: {joined}",
                lang=lang,
            )
    selected_files = parse_target_files_arg(getattr(args, "target_file", None))
    if args.target_file and not selected_files:
        if is_ko:
            exit_with_error("--target-file 값이 비어 있습니다.", lang=lang)
        else:
            exit_with_error("--target-file values are empty.", lang=lang)
    excluded_exts = parse_exclude_exts_arg(getattr(args, "exclude_ext", None))
    if args.exclude_ext and not excluded_exts:
        if is_ko:
            exit_with_error("--exclude-ext 값이 비어 있습니다.", lang=lang)
        else:
            exit_with_error("--exclude-ext values are empty.", lang=lang)
    if args.charset:
        charset_resolved = resolve_charset_source(args.charset, get_script_dir())
        if is_ko:
            _log_info(f"SDF 생성 글자셋: {charset_resolved}")
        else:
            _log_info(f"SDF generation charset: {charset_resolved}")

    if args.split_save_force and args.oneshot_save_force:
        if is_ko:
            exit_with_error(
                "--split-save-force와 --oneshot-save-force를 동시에 사용할 수 없습니다.",
                lang=lang,
            )
        else:
            exit_with_error(
                "Cannot use --split-save-force and --oneshot-save-force at the same time.",
                lang=lang,
            )

    # KR: 기본은 split-save 폴백을 활성화합니다.
    # EN: By default, enable split-save fallback.
    args.split_save = not args.oneshot_save_force
    if args.scan_jobs < 1:
        if is_ko:
            exit_with_error("--scan-jobs는 1 이상의 정수여야 합니다.", lang=lang)
        else:
            exit_with_error(
                "--scan-jobs must be an integer greater than or equal to 1.", lang=lang
            )
    if args.scan_stall_seconds < 0:
        if is_ko:
            exit_with_error(
                "--scan-stall-seconds는 0 이상의 숫자여야 합니다.",
                lang=lang,
            )
        else:
            exit_with_error(
                "--scan-stall-seconds must be greater than or equal to 0.",
                lang=lang,
            )
    if args.outline_ratio <= 0:
        if is_ko:
            exit_with_error(
                "--outline-ratio는 0보다 큰 실수여야 합니다.",
                lang=lang,
            )
        else:
            exit_with_error(
                "--outline-ratio must be a float greater than 0.",
                lang=lang,
            )
    interactive_mode_requested = len(explicit_primary_modes) == 0
    scan_jobs_explicit = any(
        arg == "--scan-jobs"
        or arg == "--max-workers"
        or arg.startswith("--scan-jobs=")
        or arg.startswith("--max-workers=")
        for arg in sys.argv[1:]
    )

    if args._scan_worker_server:
        if not args.gamepath:
            _log_console("[scan_worker] --gamepath is required.")
            raise SystemExit(2)
        worker_lang = cast(
            Language,
            args._scan_worker_lang if args._scan_worker_lang else lang,
        )
        raise SystemExit(
            run_persistent_scan_worker(
                args.gamepath,
                lang=worker_lang,
                detect_ps5_swizzle=args.ps5_swizzle,
                scan_ttf=not args._scan_sdf_only,
                scan_sdf=not args._scan_ttf_only,
            )
        )

    if args._scan_file_worker:
        if not args.gamepath:
            if is_ko:
                _log_console("[scan_worker] 오류: --gamepath가 필요합니다.")
            else:
                _log_console("[scan_worker] Error: --gamepath is required.")
            raise SystemExit(2)
        if not args._scan_file_worker_output:
            if is_ko:
                _log_console(
                    "[scan_worker] 오류: --_scan-file-worker-output 경로가 필요합니다."
                )
            else:
                _log_console(
                    "[scan_worker] Error: --_scan-file-worker-output path is required."
                )
            raise SystemExit(2)
        raise SystemExit(
            run_scan_file_worker(
                args.gamepath,
                args._scan_file_worker,
                args._scan_file_worker_output,
                lang=lang,
                detect_ps5_swizzle=args.ps5_swizzle,
                scan_ttf=not args._scan_sdf_only,
                scan_sdf=not args._scan_ttf_only,
            )
        )

    if args.temp_dir:
        args.temp_dir = os.path.abspath(str(args.temp_dir))
        try:
            os.makedirs(args.temp_dir, exist_ok=True)
        except Exception as e:
            if is_ko:
                exit_with_error(
                    f"임시 폴더를 만들 수 없습니다: {args.temp_dir} ({e})", lang=lang
                )
            else:
                exit_with_error(
                    f"Failed to create temp directory: {args.temp_dir} ({e})", lang=lang
                )
        if is_ko:
            _log_console(f"임시 저장 경로: {args.temp_dir}")
        else:
            _log_console(f"Temp save path: {args.temp_dir}")
        register_temp_dir_for_cleanup(
            os.path.join(args.temp_dir, "unity_font_replacer_temp")
        )

    output_only_root: str | None = (
        os.path.abspath(str(args.output_only)) if args.output_only else None
    )
    preview_root: str | None = None

    if args.use_game_line_metrics:
        if is_ko:
            _log_console("줄 간격 메트릭 모드: 게임 원본 줄 간격 메트릭을 사용합니다.")
        else:
            _log_console("Line metrics mode: using original in-game line metrics.")
    else:
        if is_ko:
            _log_console(
                "줄 간격 메트릭 모드: 교체 폰트 메트릭 보정을 기본 적용합니다."
            )
        else:
            _log_console(
                "Line metrics mode: using adjusted replacement font metrics by default."
            )

    if args.use_game_material:
        if is_ko:
            _log_console("Material 모드: 게임 원본 Material 파라미터를 사용합니다.")
        else:
            _log_console("Material mode: using original in-game Material parameters.")
    else:
        if is_ko:
            _log_console(
                "Material 모드: 게임 원본 Material 스타일을 유지하고 atlas/padding 차이를 자동 보정합니다."
            )
        else:
            _log_console(
                "Material mode: preserving original in-game Material style with automatic atlas/padding correction."
            )
    if args.force_raster:
        if is_ko:
            _log_console(
                "Raster 강제 모드: SDF 교체를 Raster 기준으로 처리합니다 (렌더 모드 + Material 효과값 보정)."
            )
        else:
            _log_console(
                "Forced Raster mode: processing SDF replacements with Raster behavior (render mode + material effect neutralization)."
            )
    if args.ps5_swizzle:
        if is_ko:
            _log_console(
                "PS5 swizzle 모드: 대상 Atlas swizzle을 자동 판별해 교체 Atlas를 변환합니다 "
                f"(마스크는 텍스처 크기에 따라 자동 계산, rotate={PS5_SWIZZLE_ROTATE})."
            )
        else:
            _log_console(
                "PS5 swizzle mode: auto-detecting target atlas swizzle state and transforming replacement atlas "
                f"(masks computed per texture size, rotate={PS5_SWIZZLE_ROTATE})."
            )
    else:
        if is_ko:
            _log_console("PS5 swizzle 모드: 비활성화")
        else:
            _log_console("PS5 swizzle mode: disabled")
    if args.outline_ratio != 1.0:
        if is_ko:
            _log_console(
                f"외곽선 비율 모드: Material _OutlineWidth/_OutlineSoftness에 x{args.outline_ratio:.3f} 배율을 적용합니다."
            )
        else:
            _log_console(
                f"Outline ratio mode: applying x{args.outline_ratio:.3f} to Material _OutlineWidth/_OutlineSoftness."
            )

    if args._validate_bundle:
        raise SystemExit(
            run_validation_worker(
                args._validate_bundle,
                lang=lang,
                inner_names=args._validate_inner_name,
            )
        )

    input_path = strip_wrapping_quotes_repeated(args.gamepath) if args.gamepath else ""
    _log_debug(f"[runtime] requested_gamepath={input_path!r}")
    if not input_path:
        while True:
            if is_ko:
                entered_path = input("게임 경로를 입력하세요: ").strip()
            else:
                entered_path = input("Enter game path: ").strip()
            input_path = strip_wrapping_quotes_repeated(entered_path)
            if not input_path:
                if is_ko:
                    _log_console("게임 경로가 필요합니다. 다시 입력해주세요.")
                else:
                    _log_console("Game path is required. Please try again.")
                continue
            if not os.path.isdir(input_path):
                if is_ko:
                    _log_console(
                        f"'{input_path}'는 유효한 디렉토리가 아닙니다. 다시 입력해주세요."
                    )
                else:
                    _log_console(
                        f"'{input_path}' is not a valid directory. Please try again."
                    )
                continue
            try:
                game_path, data_path = resolve_game_path(input_path, lang=lang)
            except FileNotFoundError as e:
                if is_ko:
                    _log_console(f"{e}\n다시 입력해주세요.")
                else:
                    _log_console(f"{e}\nPlease try again.")
                continue
            break
    else:
        if not os.path.isdir(input_path):
            if is_ko:
                exit_with_error(
                    f"'{input_path}'는 유효한 디렉토리가 아닙니다.", lang=lang
                )
            else:
                exit_with_error(f"'{input_path}' is not a valid directory.", lang=lang)
        try:
            game_path, data_path = resolve_game_path(input_path, lang=lang)
        except FileNotFoundError as e:
            exit_with_error(str(e), lang=lang)

    replacements: dict[str, JsonDict] | None = None
    mode: str | None = None
    interactive_session = False
    if args.parse:
        mode = "parse"
    elif args.mulmaru:
        mode = "mulmaru"
    elif args.nanumgothic:
        mode = "nanumgothic"
    elif args.font:
        mode = "font"
    elif args.list:
        mode = "list"
    elif args.preview_export:
        mode = "preview_export"
    else:
        interactive_session = True
        if is_ko:
            while True:
                _log_console("작업을 선택하세요:")
                _log_console("  1. 폰트 정보 추출 (JSON 파일 생성)")
                _log_console("  2. JSON 파일로 폰트 교체")
                _log_console("  3. Mulmaru(물마루체)로 일괄 교체")
                _log_console("  4. NanumGothic(나눔고딕)으로 일괄 교체")
                _log_console("  5. Preview export (Atlas/Glyph crop 추출)")
                _log_console()
                choice = input("선택 (1-5): ").strip()
                if choice in {"1", "2", "3", "4", "5"}:
                    break
                _log_console("잘못된 선택입니다. 다시 입력해주세요.")
        else:
            while True:
                _log_console("Select a task:")
                _log_console("  1. Export font info (create JSON)")
                _log_console("  2. Replace fonts using JSON")
                _log_console("  3. Bulk replace with Mulmaru")
                _log_console("  4. Bulk replace with NanumGothic")
                _log_console("  5. Preview export (Atlas/Glyph crops)")
                _log_console()
                choice = input("Choose (1-5): ").strip()
                if choice in {"1", "2", "3", "4", "5"}:
                    break
                _log_console("Invalid selection. Please try again.")

        if choice == "1":
            mode = "parse"
        elif choice == "2":
            mode = "list"
            while True:
                if is_ko:
                    entered = input("JSON 파일 경로를 입력하세요: ").strip()
                else:
                    entered = input("Enter JSON file path: ").strip()
                entered = strip_wrapping_quotes_repeated(entered)
                if not entered:
                    if is_ko:
                        _log_console("JSON 파일 경로가 필요합니다. 다시 입력해주세요.")
                    else:
                        _log_console("JSON file path is required. Please try again.")
                    continue
                if os.path.exists(entered):
                    args.list = entered
                    break
                if is_ko:
                    _log_console(f"파일을 찾을 수 없습니다: '{entered}'")
                else:
                    _log_console(f"File not found: '{entered}'")
        elif choice == "3":
            mode = "mulmaru"
        elif choice == "4":
            mode = "nanumgothic"
        elif choice == "5":
            mode = "preview_export"

    args.preview_export = mode == "preview_export"

    if output_only_root and mode == "preview_export":
        if is_ko:
            exit_with_error(
                "--output-only는 --preview-export와 함께 사용할 수 없습니다.",
                lang=lang,
            )
        else:
            exit_with_error(
                "--output-only cannot be used with --preview-export.",
                lang=lang,
            )

    if output_only_root:
        try:
            os.makedirs(output_only_root, exist_ok=True)
        except Exception as e:
            if is_ko:
                exit_with_error(
                    f"출력 폴더를 만들 수 없습니다: {output_only_root} ({e})",
                    lang=lang,
                )
            else:
                exit_with_error(
                    f"Failed to create output folder: {output_only_root} ({e})",
                    lang=lang,
                )
        if is_ko:
            _log_console(
                f"출력 전용 모드: 수정 파일을 '{output_only_root}'에 저장합니다."
            )
        else:
            _log_console(
                f"Output-only mode: writing modified files to '{output_only_root}'."
            )

    if mode == "preview_export":
        preview_root = os.path.join(get_script_dir(), "preview")
        try:
            os.makedirs(preview_root, exist_ok=True)
        except Exception as e:
            if is_ko:
                exit_with_error(
                    f"preview 폴더를 만들 수 없습니다: {preview_root} ({e})",
                    lang=lang,
                )
            else:
                exit_with_error(
                    f"Failed to create preview folder: {preview_root} ({e})",
                    lang=lang,
                )
        if is_ko:
            _log_console(f"Preview 모드: '{preview_root}'에 미리보기를 저장합니다.")
        else:
            _log_console(f"Preview mode: saving previews to '{preview_root}'.")
        if args.ps5_swizzle:
            if is_ko:
                _log_console(
                    "  PS5 swizzle 활성화: preview를 unswizzle 기준으로 저장합니다."
                )
            else:
                _log_console(
                    "  PS5 swizzle enabled: saving previews in unswizzled view."
                )

    if interactive_mode_requested and not scan_jobs_explicit and _mode_uses_scan_jobs(mode):
        while True:
            if is_ko:
                entered_workers = input(
                    f"스캔 워커 수를 입력하세요 (기본 {args.scan_jobs}): "
                ).strip()
            else:
                entered_workers = input(
                    f"Enter scan worker count (default {args.scan_jobs}): "
                ).strip()
            if not entered_workers:
                break
            try:
                parsed_workers = int(entered_workers)
            except (TypeError, ValueError):
                if is_ko:
                    _log_console("숫자를 입력해주세요. (1 이상의 정수)")
                else:
                    _log_console("Please enter a number. (integer >= 1)")
                continue
            if parsed_workers < 1:
                if is_ko:
                    _log_console("스캔 워커 수는 1 이상이어야 합니다.")
                else:
                    _log_console("Scan worker count must be >= 1.")
                continue
            args.scan_jobs = parsed_workers
            break

    compile_method = get_compile_method(data_path)
    detected_unity_version = get_unity_version(game_path, lang=lang)
    default_temp_root = register_temp_dir_for_cleanup(os.path.join(data_path, "temp"))
    if os.path.exists(default_temp_root):
        shutil.rmtree(default_temp_root)

    replace_ttf = not args.sdfonly
    replace_sdf = not args.ttfonly
    material_scale_by_padding = not args.use_game_material
    font_mode_builtin = (
        mode == "font"
        and normalize_font_name(str(getattr(args, "font", ""))).strip().lower()
        in {"mulmaru", "nanumgothic"}
    )
    prefer_builtin_padding_variants = mode in {"mulmaru", "nanumgothic"} or font_mode_builtin
    if args.sdfonly and args.ttfonly:
        if is_ko:
            exit_with_error(
                "--sdfonly와 --ttfonly를 동시에 사용할 수 없습니다.", lang=lang
            )
        else:
            exit_with_error(
                "Cannot use --sdfonly and --ttfonly at the same time.", lang=lang
            )

    if is_ko:
        _log_console(f"게임 경로: {game_path}")
        _log_console(f"데이터 경로: {data_path}")
        _log_console(f"컴파일 방식: {compile_method}")
        _log_console(f"스캔 워커 수: {args.scan_jobs}")
        _log_console(
            f"스캔 무활동 정지 판정: {args.scan_stall_seconds:g}초 "
            "(0=비활성화, 총 처리시간 제한 아님)"
        )
    else:
        _log_console(f"Game path: {game_path}")
        _log_console(f"Data path: {data_path}")
        _log_console(f"Compile method: {compile_method}")
        _log_console(f"Scan workers: {args.scan_jobs}")
        _log_console(
            f"Scan inactivity stall threshold: {args.scan_stall_seconds:g}s "
            "(0=disabled, not a total runtime limit)"
        )
    _log_debug(
        f"[runtime] input_path={input_path} game_path={game_path} data_path={data_path} "
        f"compile_method={compile_method} scan_jobs={args.scan_jobs} "
        f"scan_stall_seconds={args.scan_stall_seconds} "
        f"ps5_swizzle={args.ps5_swizzle} preview_export={args.preview_export}"
    )
    _log_debug(f"[runtime] unity_version={detected_unity_version}")

    if selected_files:
        target_text = ", ".join(sorted(selected_files))
        if is_ko:
            _log_console(f"--target-file 적용: {target_text}")
        else:
            _log_console(f"Applied --target-file: {target_text}")
        _log_debug(f"[runtime] target_files={target_text}")
    if excluded_exts:
        excluded_text = ", ".join(sorted(excluded_exts))
        if is_ko:
            _log_console(f"--exclude-ext 적용: {excluded_text}")
        else:
            _log_console(f"Applied --exclude-ext: {excluded_text}")
        _log_debug(f"[runtime] exclude_exts={excluded_text}")

    _log_debug(
        f"[runtime] mode={mode} interactive={interactive_session} "
        f"replace_ttf={replace_ttf} replace_sdf={replace_sdf}"
    )

    if replace_sdf and compile_method == "Il2cpp" and not os.path.exists(
        os.path.join(data_path, "Managed")
    ):
        binary_path = os.path.join(game_path, "GameAssembly.dll")
        metadata_path = os.path.join(
            data_path, "il2cpp_data", "Metadata", "global-metadata.dat"
        )
        if not os.path.exists(binary_path) or not os.path.exists(metadata_path):
            if is_ko:
                exit_with_error(
                    "Il2cpp 게임의 경우 'Managed' 폴더 또는 'GameAssembly.dll'과 'global-metadata.dat' 파일이 필요합니다.\n올바른 Unity 게임 폴더인지 확인해주세요.",
                    lang=lang,
                )
            else:
                exit_with_error(
                    "For Il2cpp games, the 'Managed' folder or 'GameAssembly.dll' and 'global-metadata.dat' files are required.\nPlease check that this is a valid Unity game folder.",
                    lang=lang,
                )

        dumper_path = os.path.join(get_script_dir(), "Il2CppDumper", "Il2CppDumper.exe")
        target_path = os.path.join(data_path, "Managed_")
        os.makedirs(target_path, exist_ok=True)
        command = [
            os.path.abspath(dumper_path),
            os.path.abspath(binary_path),
            os.path.abspath(metadata_path),
            os.path.abspath(target_path),
        ]
        if is_ko:
            _log_console("Il2cpp 게임을 위한 Managed 폴더를 생성합니다...")
        else:
            _log_console("Creating Managed folder for Il2cpp game...")
        _log_console(os.path.abspath(target_path))

        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                startupinfo=startupinfo,
                encoding="utf-8",
            )
            if process.returncode == 0:
                _log_console(process.stdout)
                shutil.move(
                    os.path.join(data_path, "Managed_", "DummyDll"),
                    os.path.join(data_path, "Managed"),
                )
                shutil.rmtree(os.path.join(data_path, "Managed_"))
                if is_ko:
                    _log_console("더미 DLL 생성에 성공했습니다!")
                else:
                    _log_console("Dummy DLL generated successfully!")
                compile_method = get_compile_method(data_path)
                if is_ko:
                    _log_console(f"컴파일 방식 재감지: {compile_method}")
                else:
                    _log_console(f"Compile method re-detected: {compile_method}")
            else:
                _log_console(process.stderr)
                if is_ko:
                    exit_with_error("Il2cpp 더미 DLL 생성 실패", lang=lang)
                else:
                    exit_with_error("Failed to generate Il2cpp dummy DLL", lang=lang)
        except Exception as e:
            if is_ko:
                exit_with_error(f"Il2CppDumper 실행 중 예외 발생: {e}", lang=lang)
            else:
                exit_with_error(f"Exception while running Il2CppDumper: {e}", lang=lang)

    if mode == "parse":
        parse_fonts(
            game_path,
            lang=lang,
            target_files=selected_files if selected_files else None,
            exclude_exts=excluded_exts if excluded_exts else None,
            scan_jobs=args.scan_jobs,
            scan_stall_seconds=args.scan_stall_seconds,
            ps5_swizzle=args.ps5_swizzle,
        )
        _pause_before_exit(lang=lang, interactive_session=interactive_session)
        return

    if mode == "preview_export":
        if is_ko:
            _log_console(
                "Preview export 모드: 모든 SDF 폰트 Atlas/Glyph crop 미리보기를 추출합니다..."
            )
        else:
            _log_console(
                "Preview export mode: exporting Atlas/Glyph crop previews for all SDF fonts..."
            )
        replacements = create_preview_export_targets(
            game_path,
            target_files=selected_files if selected_files else None,
            exclude_exts=excluded_exts if excluded_exts else None,
            scan_jobs=args.scan_jobs,
            scan_stall_seconds=args.scan_stall_seconds,
            lang=lang,
            ps5_swizzle=args.ps5_swizzle,
        )
        if not replacements:
            if is_ko:
                _log_console("Preview 대상 SDF 폰트를 찾지 못했습니다.")
            else:
                _log_console("No SDF fonts found for preview export.")
            _pause_before_exit(lang=lang, interactive_session=interactive_session)
            return
        if is_ko:
            _log_console(f"Preview 대상 SDF 폰트: {len(replacements)}개")
        else:
            _log_console(f"Preview target SDF fonts: {len(replacements)}")
    elif mode == "mulmaru":
        if is_ko:
            _log_console("Mulmaru 폰트로 일괄 교체합니다...")
        else:
            _log_console("Bulk replacing with Mulmaru...")
        replacements = create_batch_replacements(
            game_path,
            "Mulmaru",
            replace_ttf,
            replace_sdf,
            target_files=selected_files if selected_files else None,
            exclude_exts=excluded_exts if excluded_exts else None,
            scan_jobs=args.scan_jobs,
            scan_stall_seconds=args.scan_stall_seconds,
            lang=lang,
            ps5_swizzle=args.ps5_swizzle,
        )
        ttf_count = sum(1 for v in replacements.values() if v["Type"] == "TTF")
        sdf_count = sum(1 for v in replacements.values() if v["Type"] == "SDF")
        if is_ko:
            _log_console(f"발견된 폰트: TTF {ttf_count}개, SDF {sdf_count}개")
        else:
            _log_console(f"Found fonts: TTF {ttf_count}, SDF {sdf_count}")
    elif mode == "nanumgothic":
        if is_ko:
            _log_console("NanumGothic 폰트로 일괄 교체합니다...")
        else:
            _log_console("Bulk replacing with NanumGothic...")
        replacements = create_batch_replacements(
            game_path,
            "NanumGothic",
            replace_ttf,
            replace_sdf,
            target_files=selected_files if selected_files else None,
            exclude_exts=excluded_exts if excluded_exts else None,
            scan_jobs=args.scan_jobs,
            scan_stall_seconds=args.scan_stall_seconds,
            lang=lang,
            ps5_swizzle=args.ps5_swizzle,
        )
        ttf_count = sum(1 for v in replacements.values() if v["Type"] == "TTF")
        sdf_count = sum(1 for v in replacements.values() if v["Type"] == "SDF")
        if is_ko:
            _log_console(f"발견된 폰트: TTF {ttf_count}개, SDF {sdf_count}개")
        else:
            _log_console(f"Found fonts: TTF {ttf_count}, SDF {sdf_count}")
    elif mode == "font":
        if is_ko:
            _log_console(f"{args.font} 폰트로 일괄 교체합니다...")
        else:
            _log_console(f"Bulk replacing with {args.font}...")
        replacements = create_batch_replacements(
            game_path,
            str(args.font),
            replace_ttf,
            replace_sdf,
            target_files=selected_files if selected_files else None,
            exclude_exts=excluded_exts if excluded_exts else None,
            scan_jobs=args.scan_jobs,
            scan_stall_seconds=args.scan_stall_seconds,
            lang=lang,
            ps5_swizzle=args.ps5_swizzle,
        )
        ttf_count = sum(1 for v in replacements.values() if v["Type"] == "TTF")
        sdf_count = sum(1 for v in replacements.values() if v["Type"] == "SDF")
        if is_ko:
            _log_console(f"발견된 폰트: TTF {ttf_count}개, SDF {sdf_count}개")
        else:
            _log_console(f"Found fonts: TTF {ttf_count}, SDF {sdf_count}")
    elif mode == "list":
        if isinstance(args.list, str):
            args.list = strip_wrapping_quotes_repeated(args.list)

        if interactive_session:
            while not args.list or not os.path.exists(args.list):
                if args.list:
                    if is_ko:
                        _log_console(f"'{args.list}' 파일을 찾을 수 없습니다.")
                    else:
                        _log_console(f"File not found: '{args.list}'")
                if is_ko:
                    entered = input("JSON 파일 경로를 다시 입력하세요: ").strip()
                else:
                    entered = input("Re-enter JSON file path: ").strip()
                args.list = strip_wrapping_quotes_repeated(entered)

        if not args.list or not os.path.exists(args.list):
            if is_ko:
                exit_with_error(f"'{args.list}' 파일을 찾을 수 없습니다.", lang=lang)
            else:
                exit_with_error(f"File not found: '{args.list}'", lang=lang)

        if is_ko:
            _log_console(f"'{args.list}' 파일을 읽어서 교체합니다...")
        else:
            _log_console(f"Replacing using '{args.list}'...")
        try:
            replacements = load_replacement_mapping_file(args.list)
        except ValueError:
            if is_ko:
                exit_with_error("JSON 루트는 객체(dict)여야 합니다.", lang=lang)
            else:
                exit_with_error("JSON root must be an object (dict).", lang=lang)

    if replacements is None:
        if is_ko:
            exit_with_error("교체 정보가 생성되지 않았습니다.", lang=lang)
        else:
            exit_with_error("Replacement mapping was not generated.", lang=lang)

    if selected_files:
        replacements = {
            key: value
            for key, value in replacements.items()
            if isinstance(value, dict)
            and os.path.basename(str(value.get("File", ""))) in selected_files
        }

        if not replacements:
            target_text = ", ".join(sorted(selected_files))
            if is_ko:
                exit_with_error(
                    f"--target-file 조건에 맞는 교체 대상이 없습니다: {target_text}",
                    lang=lang,
                )
            else:
                exit_with_error(
                    f"No replacement targets matched --target-file: {target_text}",
                    lang=lang,
                )

    if mode != "preview_export":
        _ensure_custom_unitypy_streaming_save(lang=lang)

    unity_version = detected_unity_version
    generator = (
        _create_generator(
            unity_version,
            game_path,
            data_path,
            compile_method,
            lang=lang,
        )
        if replace_sdf
        else None
    )
    replacement_lookup, files_to_process = build_replacement_lookup(replacements)
    _log_debug(
        f"[runtime] replacement_entries={len(replacements)} "
        f"lookup_entries={len(replacement_lookup)} files_to_process={len(files_to_process)}"
    )
    preview_files_to_process: set[str] = set()
    if args.preview_export:
        preview_files_to_process = {
            os.path.basename(str(value.get("File", "")))
            for value in replacements.values()
            if isinstance(value, dict) and str(value.get("Type", "")) == "SDF"
        }
        preview_files_to_process.discard("")
    process_files = set(files_to_process) | preview_files_to_process
    _log_debug(
        f"[runtime] process_files={len(process_files)} "
        f"preview_only_files={len(preview_files_to_process)}"
    )
    all_assets_files = find_assets_files(
        game_path,
        lang=lang,
        exclude_exts=excluded_exts if excluded_exts else None,
    )
    asset_file_index = _build_asset_file_index(all_assets_files, data_path)
    asset_path_by_key = cast(dict[str, str], asset_file_index.get("path_by_key", {}))
    basename_to_keys = cast(
        dict[str, list[str]],
        asset_file_index.get("basename_to_keys", {}),
    )
    duplicate_asset_names: dict[str, list[str]] = {
        basename: [asset_path_by_key[key] for key in keys if key in asset_path_by_key]
        for basename, keys in basename_to_keys.items()
        if len(keys) > 1
    }
    if duplicate_asset_names:
        for duplicate_name, duplicate_paths in sorted(duplicate_asset_names.items()):
            _log_warning(
                f"[runtime] duplicate_asset_basename={duplicate_name} "
                f"count={len(duplicate_paths)} paths={duplicate_paths}"
            )
    asset_file_queue: list[str] = [
        asset_key
        for asset_key, asset_path in asset_path_by_key.items()
        if os.path.basename(asset_path) in process_files
    ]
    matched_process_files = {
        os.path.basename(asset_path_by_key[key])
        for key in asset_file_queue
        if key in asset_path_by_key
    }
    missing_process_files = sorted(process_files - matched_process_files)
    if missing_process_files:
        raise FileNotFoundError(
            "Replacement target asset file(s) were not found or are not safely writable: "
            + ", ".join(missing_process_files)
        )
    _log_debug(
        f"[runtime] matched_asset_files={len(asset_file_queue)} all_candidates={len(all_assets_files)}"
    )
    deferred_transaction = (
        _DeferredPatchTransaction(
            os.path.join(
                args.temp_dir or game_path,
                ".unity_font_replacer_rollback",
            )
        )
        if mode != "preview_export"
        else None
    )
    _ACTIVE_DEFERRED_TRANSACTION.set(deferred_transaction)
    if output_only_root and mode != "preview_export":
        prepare_output_only_dependencies(
            data_path,
            output_only_root,
            lang=lang,
            transaction=deferred_transaction,
        )

    deferred_texture_plans: dict[str, dict[str, Any]] = {}
    deferred_material_plans: dict[str, dict[str, Any]] = {}
    deferred_material_atlas_plans: dict[str, dict[str, Any]] = {}
    collected_material_atlas_plans: dict[str, JsonDict] = {}
    pending_external_patch_files: set[str] = set()
    pending_queue_keys: set[str] = set(asset_file_queue)
    prepared_output_targets: set[str] = set()
    terminal_failures: list[str] = []
    modified_count = 0
    queue_index = 0
    while queue_index < len(asset_file_queue):
        asset_file_key = asset_file_queue[queue_index]
        queue_index += 1
        pending_queue_keys.discard(asset_file_key)
        assets_file = asset_path_by_key.get(asset_file_key)
        if not assets_file:
            _log_warning(f"[runtime] queued file not found: {asset_file_key}")
            continue
        fn = os.path.basename(assets_file)
        working_assets_file = assets_file
        if output_only_root and mode != "preview_export":
            working_assets_file = resolve_output_only_path(
                assets_file, data_path, output_only_root
            )
            working_dir = os.path.dirname(working_assets_file)
            if working_dir and not os.path.exists(working_dir):
                os.makedirs(working_dir, exist_ok=True)
            working_assets_key = (
                _normalize_asset_file_key(working_assets_file) or working_assets_file
            )
            if working_assets_key not in prepared_output_targets:
                if deferred_transaction is not None:
                    deferred_transaction.backup(
                        working_assets_file, allow_missing=True
                    )
                shutil.copy2(assets_file, working_assets_file)
                prepared_output_targets.add(working_assets_key)
                if is_ko:
                    rel_out = os.path.relpath(working_assets_file, output_only_root)
                    _log_console(f"  출력 대상 준비: {rel_out}")
                else:
                    rel_out = os.path.relpath(working_assets_file, output_only_root)
                    _log_console(f"  Prepared output target: {rel_out}")
        if (
            fn in process_files
            or asset_file_key in deferred_texture_plans
            or asset_file_key in deferred_material_plans
            or asset_file_key in deferred_material_atlas_plans
        ):
            if is_ko:
                _log_console(f"\n처리 중: {fn}")
            else:
                _log_console(f"\nProcessing: {fn}")
            # KR: 기본은 split-save 폴백을 사용하고, --oneshot-save-force일 때만 비활성화합니다.
            # EN: By default, use split-save fallback; only disable when --oneshot-save-force is set.
            file_replacements = {
                key: value
                for key, value in replacements.items()
                if isinstance(value, dict)
                and value.get("File") == fn
                and value.get("Replace_to")
            }
            file_ttf_replacements = {
                key: value
                for key, value in file_replacements.items()
                if value.get("Type") == "TTF"
            }
            file_sdf_replacements = {
                key: value
                for key, value in file_replacements.items()
                if value.get("Type") == "SDF"
            }
            _log_replacement_plan_details(fn, file_replacements)

            file_modified = False
            use_split_sdf_save = (
                args.split_save and replace_sdf and len(file_sdf_replacements) > 1
            )

            if use_split_sdf_save:
                if is_ko:
                    _log_console(
                        f"  SDF 대상 {len(file_sdf_replacements)}건: one-shot 실패 시 적응형 분할 저장으로 폴백합니다..."
                    )
                else:
                    _log_console(
                        f"  {len(file_sdf_replacements)} SDF targets: will fall back to adaptive split save if one-shot fails..."
                    )

                # KR: 먼저 한 번에 저장을 시도하고, 실패 시에만 적응형 분할 저장으로 폴백합니다.
                # EN: First attempt a one-shot save; fall back to adaptive split save only on failure.
                file_lookup, _ = build_replacement_lookup(file_replacements)
                one_shot_ok = False
                one_shot_outcome: JsonDict = {}
                if args.split_save_force:
                    if is_ko:
                        _log_console(
                            "  --split-save-force 활성화: one-shot을 건너뛰고 SDF 1개씩 강제 분할 저장을 시작합니다..."
                        )
                    else:
                        _log_console(
                            "  --split-save-force enabled: skipping one-shot and forcing one-by-one SDF split save..."
                        )
                else:
                    try:
                        one_shot_ok = replace_fonts_in_file(
                            unity_version,
                            game_path,
                            working_assets_file,
                            file_replacements,
                            replace_ttf=replace_ttf,
                            replace_sdf=replace_sdf,
                            use_game_mat=args.use_game_material,
                            force_raster=args.force_raster,
                            use_game_line_metrics=args.use_game_line_metrics,
                            material_scale_by_padding=material_scale_by_padding,
                            outline_ratio=args.outline_ratio,
                            prefer_original_compress=args.original_compress,
                            temp_root_dir=args.temp_dir,
                            generator=generator,
                            replacement_lookup=file_lookup,
                            ps5_swizzle=args.ps5_swizzle,
                            preview_export=args.preview_export,
                            preview_root=preview_root,
                            prefer_builtin_padding_variants=prefer_builtin_padding_variants,
                            charset_source=args.charset,
                            asset_file_index=asset_file_index,
                            deferred_texture_plans=deferred_texture_plans,
                            deferred_material_plans=deferred_material_plans,
                            deferred_material_atlas_plans=deferred_material_atlas_plans,
                            collected_material_atlas_plans=collected_material_atlas_plans,
                            pending_external_patch_files=pending_external_patch_files,
                            logical_file_key=asset_file_key,
                            deferred_transaction=deferred_transaction,
                            operation_outcome=one_shot_outcome,
                            lang=lang,
                        )
                    except MemoryError as e:
                        if is_ko:
                            _log_console(f"  one-shot 저장 실패 [MemoryError]: {e!r}")
                            _log_console("  적응형 분할 저장으로 폴백합니다...")
                        else:
                            _log_console(f"  One-shot save failed [MemoryError]: {e!r}")
                            _log_console("  Falling back to adaptive split save...")
                    except Exception as e:
                        if is_ko:
                            _log_console(
                                f"  one-shot 저장 실패 [{type(e).__name__}]: {e!r}"
                            )
                            _log_console("  적응형 분할 저장으로 폴백합니다...")
                        else:
                            _log_console(
                                f"  One-shot save failed [{type(e).__name__}]: {e!r}"
                            )
                            _log_console("  Falling back to adaptive split save...")

                if one_shot_ok:
                    file_modified = True
                elif deferred_transaction is not None and deferred_transaction.has_failures:
                    pass
                else:
                    auto_split_profile: JsonDict | None = None
                    suggested_sdf_batch_size = 0
                    split_stopped = False
                    if replace_ttf and file_ttf_replacements:
                        file_ttf_lookup, _ = build_replacement_lookup(
                            file_ttf_replacements
                        )
                        ttf_outcome: JsonDict = {}
                        try:
                            ttf_ok = replace_fonts_in_file(
                                unity_version,
                                game_path,
                                working_assets_file,
                                file_ttf_replacements,
                                replace_ttf=True,
                                replace_sdf=False,
                                use_game_mat=args.use_game_material,
                                force_raster=args.force_raster,
                                use_game_line_metrics=args.use_game_line_metrics,
                                material_scale_by_padding=material_scale_by_padding,
                                outline_ratio=args.outline_ratio,
                                prefer_original_compress=args.original_compress,
                                temp_root_dir=args.temp_dir,
                                generator=generator,
                                replacement_lookup=file_ttf_lookup,
                                ps5_swizzle=args.ps5_swizzle,
                                preview_export=args.preview_export,
                                preview_root=preview_root,
                                prefer_builtin_padding_variants=prefer_builtin_padding_variants,
                                charset_source=args.charset,
                                asset_file_index=asset_file_index,
                                deferred_texture_plans=deferred_texture_plans,
                                deferred_material_plans=deferred_material_plans,
                                deferred_material_atlas_plans=deferred_material_atlas_plans,
                                collected_material_atlas_plans=collected_material_atlas_plans,
                                pending_external_patch_files=pending_external_patch_files,
                                logical_file_key=asset_file_key,
                                deferred_transaction=deferred_transaction,
                                operation_outcome=ttf_outcome,
                                lang=lang,
                            )
                            if ttf_ok:
                                file_modified = True
                            elif not bool(ttf_outcome.get("already_satisfied")):
                                split_stopped = True
                                failure = f"{fn}: terminal TTF split save failure"
                                terminal_failures.append(failure)
                                if deferred_transaction is not None:
                                    deferred_transaction.fail(failure)
                        except Exception as e:
                            if is_ko:
                                _log_console(
                                    f"  TTF 분할 저장 실패 [{type(e).__name__}]: {e!r}"
                                )
                            else:
                                _log_console(
                                    f"  TTF split save failed [{type(e).__name__}]: {e!r}"
                                )
                            split_stopped = True
                            failure = (
                                f"{fn}: terminal TTF split exception: "
                                f"{type(e).__name__}: {e}"
                            )
                            terminal_failures.append(failure)
                            if deferred_transaction is not None:
                                deferred_transaction.fail(failure)
                        if (
                            deferred_transaction is not None
                            and deferred_transaction.has_failures
                        ):
                            split_stopped = True

                    if replace_sdf and not split_stopped:
                        if not args.split_save_force:
                            auto_split_profile = _estimate_sdf_texture_batch_profile(
                                file_sdf_replacements,
                                force_raster=args.force_raster,
                            )
                            suggested_sdf_batch_size = int(
                                auto_split_profile.get("suggested_batch_size", 0) or 0
                            )
                            estimated_texture_bytes = int(
                                auto_split_profile.get("estimated_total_bytes", 0)
                                or 0
                            )
                            estimated_texture_targets = int(
                                auto_split_profile.get("estimated_target_count", 0)
                                or 0
                            )
                            if estimated_texture_bytes > 0:
                                _log_debug(
                                    f"[split_save_estimate] file={fn} targets={estimated_texture_targets} "
                                    f"estimated_total={estimated_texture_bytes} "
                                    f"suggested_batch_size={suggested_sdf_batch_size}"
                                )
                                if suggested_sdf_batch_size > 0:
                                    if is_ko:
                                        _log_console(
                                            "  one-shot 실패 후 적응형 분할 저장 초기 배치를 "
                                            f"{suggested_sdf_batch_size}로 시작합니다 "
                                            f"(예상 texture payload: {_format_byte_size(estimated_texture_bytes)})."
                                        )
                                    else:
                                        _log_console(
                                            "  One-shot failed; starting adaptive split save with "
                                            f"initial batch {suggested_sdf_batch_size} "
                                            f"(estimated texture payload: {_format_byte_size(estimated_texture_bytes)})."
                                        )
                        sdf_items = list(file_sdf_replacements.items())
                        sdf_total = len(sdf_items)
                        if sdf_total > 0:
                            if args.split_save_force:
                                batch_size = 1
                            elif suggested_sdf_batch_size > 0:
                                batch_size = min(
                                    sdf_total,
                                    max(1, suggested_sdf_batch_size),
                                )
                            else:
                                batch_size = min(sdf_total, max(1, sdf_total // 2))

                            idx = 0
                            while idx < sdf_total:
                                current_batch = min(batch_size, sdf_total - idx)
                                batch_dict = dict(sdf_items[idx : idx + current_batch])
                                batch_lookup, _ = build_replacement_lookup(batch_dict)
                                batch_outcome: JsonDict = {}

                                try:
                                    ok = replace_fonts_in_file(
                                        unity_version,
                                        game_path,
                                        working_assets_file,
                                        batch_dict,
                                        replace_ttf=False,
                                        replace_sdf=True,
                                        use_game_mat=args.use_game_material,
                                        force_raster=args.force_raster,
                                        use_game_line_metrics=args.use_game_line_metrics,
                                        material_scale_by_padding=material_scale_by_padding,
                                        outline_ratio=args.outline_ratio,
                                        prefer_original_compress=args.original_compress,
                                        temp_root_dir=args.temp_dir,
                                        generator=generator,
                                        replacement_lookup=batch_lookup,
                                        ps5_swizzle=args.ps5_swizzle,
                                        preview_export=args.preview_export,
                                        preview_root=preview_root,
                                        prefer_builtin_padding_variants=prefer_builtin_padding_variants,
                                        charset_source=args.charset,
                                        asset_file_index=asset_file_index,
                                        deferred_texture_plans=deferred_texture_plans,
                                        deferred_material_plans=deferred_material_plans,
                                        deferred_material_atlas_plans=deferred_material_atlas_plans,
                                        collected_material_atlas_plans=collected_material_atlas_plans,
                                        pending_external_patch_files=pending_external_patch_files,
                                        logical_file_key=asset_file_key,
                                        deferred_transaction=deferred_transaction,
                                        operation_outcome=batch_outcome,
                                        lang=lang,
                                    )
                                except Exception as e:
                                    ok = False
                                    if is_ko:
                                        _log_console(
                                            f"  SDF 배치 저장 실패 [{type(e).__name__}]: {e!r}"
                                        )
                                    else:
                                        _log_console(
                                            f"  SDF batch save failed [{type(e).__name__}]: {e!r}"
                                        )

                                if not ok and bool(
                                    batch_outcome.get("already_satisfied")
                                ):
                                    ok = True

                                if ok:
                                    if not bool(
                                        batch_outcome.get("already_satisfied")
                                    ):
                                        file_modified = True
                                    idx += current_batch
                                    gc.collect()
                                    if idx < sdf_total:
                                        if args.split_save_force:
                                            if is_ko:
                                                _log_console(
                                                    f"  SDF 배치 진행: {idx}/{sdf_total} (다음 배치: 1, 강제)"
                                                )
                                            else:
                                                _log_console(
                                                    f"  SDF batch progress: {idx}/{sdf_total} (next batch: 1, forced)"
                                                )
                                        else:
                                            # KR: 성공하면 배치를 키워 쓰기 횟수를 줄입니다.
                                            # EN: On success, increase batch size to reduce the number of writes.
                                            batch_size = min(
                                                sdf_total - idx,
                                                max(
                                                    current_batch + 1, current_batch * 2
                                                ),
                                            )
                                            if is_ko:
                                                _log_console(
                                                    f"  SDF 배치 진행: {idx}/{sdf_total} (다음 배치: {batch_size})"
                                                )
                                            else:
                                                _log_console(
                                                    f"  SDF batch progress: {idx}/{sdf_total} (next batch: {batch_size})"
                                                )
                                else:
                                    if (
                                        deferred_transaction is not None
                                        and deferred_transaction.has_failures
                                    ):
                                        split_stopped = True
                                        break
                                    if is_ko:
                                        _log_console(
                                            "  SDF 배치 저장 실패: 내부 저장 단계가 False를 반환했습니다. 위 오류 로그를 확인하세요."
                                        )
                                    else:
                                        _log_console(
                                            "  SDF batch save failed: internal save stage returned False. Check previous error logs."
                                        )
                                    if current_batch <= 1:
                                        split_stopped = True
                                        failure = (
                                            f"{fn}: terminal SDF split save failure"
                                        )
                                        terminal_failures.append(failure)
                                        if deferred_transaction is not None:
                                            deferred_transaction.fail(failure)
                                        if is_ko:
                                            _log_console(
                                                "  SDF 분할 저장 중단: 배치 1개에서도 저장 실패"
                                            )
                                        else:
                                            _log_console(
                                                "  Stopping SDF split save: failed even with batch size 1"
                                            )
                                        break

                                    batch_size = max(1, current_batch // 2)
                                    gc.collect()
                                    if is_ko:
                                        _log_console(
                                            f"  SDF 배치 크기를 {batch_size}로 줄여 재시도합니다..."
                                        )
                                    else:
                                        _log_console(
                                            f"  Reducing SDF batch size to {batch_size} and retrying..."
                                        )
            else:
                if (
                    replace_sdf
                    and len(file_sdf_replacements) > 1
                    and not args.split_save
                ):
                    if is_ko:
                        _log_console(
                            "  참고: --oneshot-save-force로 split-save 폴백이 비활성화되어 메모리 피크가 증가할 수 있습니다."
                        )
                    else:
                        _log_console(
                            "  Note: --oneshot-save-force disables split-save fallback and may increase memory peak."
                        )
                direct_outcome: JsonDict = {}
                try:
                    direct_ok = replace_fonts_in_file(
                        unity_version,
                        game_path,
                        working_assets_file,
                        replacements,
                        replace_ttf,
                        replace_sdf,
                        use_game_mat=args.use_game_material,
                        force_raster=args.force_raster,
                        use_game_line_metrics=args.use_game_line_metrics,
                        material_scale_by_padding=material_scale_by_padding,
                        outline_ratio=args.outline_ratio,
                        prefer_original_compress=args.original_compress,
                        temp_root_dir=args.temp_dir,
                        generator=generator,
                        replacement_lookup=replacement_lookup,
                        ps5_swizzle=args.ps5_swizzle,
                        preview_export=args.preview_export,
                        preview_root=preview_root,
                        prefer_builtin_padding_variants=prefer_builtin_padding_variants,
                        charset_source=args.charset,
                        asset_file_index=asset_file_index,
                        deferred_texture_plans=deferred_texture_plans,
                        deferred_material_plans=deferred_material_plans,
                        deferred_material_atlas_plans=deferred_material_atlas_plans,
                        collected_material_atlas_plans=collected_material_atlas_plans,
                        pending_external_patch_files=pending_external_patch_files,
                        logical_file_key=asset_file_key,
                        deferred_transaction=deferred_transaction,
                        operation_outcome=direct_outcome,
                        lang=lang,
                    )
                    if direct_ok:
                        file_modified = True
                    elif (
                        (
                            int(direct_outcome.get("requested_targets", 0) or 0)
                            > 0
                            and not bool(direct_outcome.get("already_satisfied"))
                        )
                        or (
                            bool(direct_outcome.get("modified"))
                            and not bool(direct_outcome.get("save_success"))
                        )
                    ):
                        failure = f"{fn}: requested font save failed"
                        terminal_failures.append(failure)
                        if deferred_transaction is not None:
                            deferred_transaction.fail(failure)
                except Exception as e:
                    if is_ko:
                        _log_console(f"  파일 처리 실패 [{type(e).__name__}]: {e!r}")
                    else:
                        _log_console(
                            f"  File processing failed [{type(e).__name__}]: {e!r}"
                        )
                    failure = (
                        f"{fn}: file processing exception: "
                        f"{type(e).__name__}: {e}"
                    )
                    terminal_failures.append(failure)
                    if deferred_transaction is not None:
                        deferred_transaction.fail(failure)

            if file_modified:
                modified_count += 1

            if (
                deferred_transaction is not None
                and deferred_transaction.has_failures
            ):
                break

        if pending_external_patch_files:
            queued_from_external = sorted(pending_external_patch_files)
            pending_external_patch_files.clear()
            for pending_key in queued_from_external:
                pending_path = asset_path_by_key.get(pending_key)
                if not pending_path:
                    _log_warning(
                        f"[runtime] deferred target file not found: {pending_key}"
                    )
                    continue
                if pending_key in pending_queue_keys:
                    continue
                asset_file_queue.append(pending_key)
                pending_queue_keys.add(pending_key)
                _log_debug(
                    f"[runtime] queued_deferred_patch_file={pending_path} "
                    f"queue_size={len(asset_file_queue)}"
                )

    reconciliation_buckets = _build_material_atlas_reconciliation_buckets(
        asset_file_queue,
        collected_material_atlas_plans,
    )
    if (
        mode != "preview_export"
        and reconciliation_buckets
        and not (
            deferred_transaction is not None
            and deferred_transaction.has_failures
        )
    ):
        if is_ko:
            _log_console("\n외부 아틀라스 머티리얼 프리셋 재검사 중...")
        else:
            _log_console("\nReconciling external-atlas Material presets...")
        for asset_file_key in list(reconciliation_buckets):
            assets_file = asset_path_by_key.get(asset_file_key)
            if not assets_file:
                continue
            working_assets_file = assets_file
            if output_only_root:
                working_assets_file = resolve_output_only_path(
                    assets_file,
                    data_path,
                    output_only_root,
                )
            reconciliation_outcome: JsonDict = {}
            try:
                replace_fonts_in_file(
                    unity_version,
                    game_path,
                    working_assets_file,
                    {},
                    replace_ttf=False,
                    replace_sdf=False,
                    use_game_mat=args.use_game_material,
                    force_raster=args.force_raster,
                    use_game_line_metrics=args.use_game_line_metrics,
                    material_scale_by_padding=material_scale_by_padding,
                    outline_ratio=args.outline_ratio,
                    prefer_original_compress=args.original_compress,
                    temp_root_dir=args.temp_dir,
                    generator=generator,
                    replacement_lookup={},
                    ps5_swizzle=args.ps5_swizzle,
                    prefer_builtin_padding_variants=prefer_builtin_padding_variants,
                    charset_source=args.charset,
                    asset_file_index=asset_file_index,
                    deferred_material_atlas_plans=reconciliation_buckets,
                    logical_file_key=asset_file_key,
                    deferred_transaction=deferred_transaction,
                    operation_outcome=reconciliation_outcome,
                    lang=lang,
                )
                if bool(reconciliation_outcome.get("modified")) and not bool(
                    reconciliation_outcome.get("save_success")
                ):
                    failure = (
                        f"{os.path.basename(assets_file)}: "
                        "material reconciliation save failed"
                    )
                    terminal_failures.append(failure)
                    if deferred_transaction is not None:
                        deferred_transaction.fail(failure)
            except Exception as exc:
                failure = (
                    f"{os.path.basename(assets_file)}: material reconciliation "
                    f"exception: {type(exc).__name__}: {exc}"
                )
                terminal_failures.append(failure)
                if deferred_transaction is not None:
                    deferred_transaction.fail(failure)
            if (
                deferred_transaction is not None
                and deferred_transaction.has_failures
            ):
                break

    for remaining_bucket in reconciliation_buckets.values():
        _cleanup_deferred_patch_bucket(remaining_bucket)
    reconciliation_buckets.clear()
    _cleanup_deferred_patch_bucket(collected_material_atlas_plans)
    collected_material_atlas_plans.clear()

    pending_required_deferred = bool(
        deferred_texture_plans
        or deferred_material_plans
        or (
            deferred_transaction is not None
            and deferred_transaction.has_failures
        )
    )
    deferred_failure_message: str | None = None
    if deferred_transaction is not None:
        if pending_required_deferred:
            rollback_count = deferred_transaction.backup_count
            rollback_directory = deferred_transaction.backup_directory
            rollback_ok = deferred_transaction.rollback()
            modified_count = max(0, modified_count - rollback_count)
            message = (
                "요청한 폰트/TMP 패치가 완결되지 않아 관련 파일을 원래 상태로 되돌렸습니다."
                if is_ko
                else "A requested font/TMP patch was incomplete; related files were rolled back."
            )
            _log_console(f"\n오류: {message}" if is_ko else f"\nError: {message}")
            deferred_failure_message = message
            if not rollback_ok:
                deferred_failure_message = (
                    "외부 TMP 패치 롤백에 실패했습니다. "
                    f"로그와 rollback 백업을 확인하세요: {rollback_directory}"
                    if is_ko
                    else "Failed to roll back an external TMP patch. "
                    f"Check the log and rollback backups: {rollback_directory}"
                )
        else:
            deferred_transaction.commit()

    for deferred_map in (
        deferred_texture_plans,
        deferred_material_plans,
        deferred_material_atlas_plans,
    ):
        for remaining_bucket in deferred_map.values():
            _cleanup_deferred_patch_bucket(remaining_bucket)
        deferred_map.clear()

    if deferred_failure_message:
        raise DeferredPatchAtomicityError(deferred_failure_message)
    if terminal_failures:
        raise RuntimeError("; ".join(terminal_failures))

    if mode == "preview_export":
        if is_ko:
            _log_console(
                f"\n완료! preview export 처리 파일: {len(process_files)}개 (원본 수정 없음)"
            )
        else:
            _log_console(
                f"\nDone! Preview-export processed {len(process_files)} file(s) (no source modifications)."
            )
        _pause_before_exit(lang=lang, interactive_session=interactive_session)
    else:
        if is_ko:
            _log_console(f"\n완료! {modified_count}개의 파일이 수정되었습니다.")
        else:
            _log_console(f"\nDone! Modified {modified_count} file(s).")
        _pause_before_exit(lang=lang, interactive_session=interactive_session)


def main() -> None:
    """KR: 한국어 CLI 진입점입니다.
    EN: Korean CLI entry point.
    """
    main_cli(lang="ko")


def main_en() -> None:
    """KR: 영어 CLI 진입점입니다.
    EN: English CLI entry point.
    """
    main_cli(lang="en")


def run_main_ko() -> None:
    """KR: 한국어 실행 진입점을 예외 처리와 함께 실행합니다.
    EN: Runs the Korean entry point with exception handling.
    """
    try:
        main()
    except Exception as e:
        _log_exception(f"\n예상치 못한 오류가 발생했습니다: {e}")
        _pause_before_exit(lang="ko", interactive_session=False)
        sys.exit(1)
    finally:
        logging.shutdown()
        cleanup_registered_temp_dirs()


def run_main_en() -> None:
    """KR: 영어 실행 진입점을 예외 처리와 함께 실행합니다.
    EN: Runs the English entry point with exception handling.
    """
    try:
        main_en()
    except Exception as e:
        _log_exception(f"\nAn unexpected error occurred: {e}")
        _pause_before_exit(lang="en", interactive_session=False)
        sys.exit(1)
    finally:
        logging.shutdown()
        cleanup_registered_temp_dirs()


if __name__ == "__main__":
    try:
        run_main_ko()
    except Exception as e:
        _log_exception(f"\n예상치 못한 오류가 발생했습니다: {e}")
        _pause_before_exit(lang="ko", interactive_session=False)
        sys.exit(1)
