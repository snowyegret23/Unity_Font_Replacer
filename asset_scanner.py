import gc
import logging
import os
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from ps5_texture import detect_texture_object_ps5_swizzle
from tmp_font_schema import inspect_tmp_font_schema
from unitypy_runtime import close_unitypy_env, load_unitypy

if TYPE_CHECKING:
    from UnityPy.helpers.TypeTreeGenerator import TypeTreeGenerator

Language = Literal["ko", "en"]
JsonDict = dict[str, Any]
_UNITY_SPLIT_FILE_RE = re.compile(r"\.split\d+$", re.IGNORECASE)
logger = logging.getLogger(__name__)


def _default_info_log(message: str) -> None:
    logger.info("%s", message)


def _default_debug_log(message: str) -> None:
    logger.debug("%s", message)


def _emit_phase_callback(
    phase_callback: Callable[[str, JsonDict], None] | None,
    phase: str,
    **payload: Any,
) -> None:
    if phase_callback is None:
        return
    try:
        phase_callback(phase, dict(payload))
    except Exception:
        logger.debug("scan phase callback failed: %s", phase, exc_info=True)


def find_assets_files(
    game_path: str,
    lang: Language = "ko",
    target_files: set[str] | None = None,
    exclude_exts: set[str] | None = None,
    *,
    data_path_resolver: Callable[..., str],
    log_console: Callable[[str], None],
) -> list[str]:
    """KR: 게임에서 처리 대상 에셋 파일 목록을 수집합니다.
    target_files가 있으면 해당 파일명으로 스캔 대상을 제한합니다.
    exclude_exts가 있으면 해당 확장자를 추가 제외합니다.
    EN: Collects the list of asset files to process from the game.
    If target_files is provided, limits scan targets to those filenames.
    If exclude_exts is provided, additionally excludes those extensions.
    """
    data_path = data_path_resolver(game_path, lang=lang)
    assets_files: list[str] = []
    skipped_split_files: list[str] = []
    normalized_targets = (
        {os.path.basename(name) for name in target_files} if target_files else None
    )
    blacklist_exts = {
        ".dll",
        ".manifest",
        ".exe",
        ".txt",
        ".json",
        ".xml",
        ".log",
        ".ini",
        ".cfg",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".wav",
        ".mp3",
        ".ogg",
        ".mp4",
        ".avi",
        ".mov",
        ".bak",
        ".info",
        ".config",
        ".browser",
        ".aspx",
        ".map",
        ".resource",
        ".resources",
        ".rollback",
        ".tmp",
    }
    if exclude_exts:
        blacklist_exts.update({str(ext).lower() for ext in exclude_exts if ext})

    skip_root_prefixes = [
        os.path.normcase(
            os.path.normpath(os.path.join(data_path, "il2cpp_data", "etc", "mono"))
        )
    ]

    normalized_data_root = os.path.normcase(os.path.normpath(data_path))
    for root, dirs, files in os.walk(data_path):
        normalized_root = os.path.normcase(os.path.normpath(root))
        excluded_tool_dirs = {".unity_font_replacer_rollback"}
        if normalized_root == normalized_data_root:
            excluded_tool_dirs.add("temp")
        dirs[:] = [
            directory
            for directory in dirs
            if directory.lower() not in excluded_tool_dirs
        ]
        if any(
            normalized_root == prefix or normalized_root.startswith(prefix + os.sep)
            for prefix in skip_root_prefixes
        ):
            dirs[:] = []
            continue
        for fn in files:
            if normalized_targets is not None and fn not in normalized_targets:
                continue
            if _UNITY_SPLIT_FILE_RE.search(fn):
                # UnityPy can merge split files for reading, but save_to() emits
                # one complete file and cannot safely reconstruct the pieces.
                skipped_split_files.append(os.path.join(root, fn))
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in blacklist_exts:
                continue
            assets_files.append(os.path.join(root, fn))
    assets_files.sort()
    if skipped_split_files:
        if lang == "ko":
            log_console(
                "경고: 안전한 재분할 저장을 지원하지 않아 Unity .splitN 파일 "
                f"{len(skipped_split_files)}개를 건너뜁니다."
            )
        else:
            log_console(
                "Warning: skipped "
                f"{len(skipped_split_files)} Unity .splitN file(s); safe split-file "
                "saving is not supported."
            )
    return assets_files


def scan_fonts_from_env(
    env: Any,
    file_name: str,
    lang: Language = "ko",
    detect_ps5_swizzle: bool = False,
    scan_ttf: bool = True,
    scan_sdf: bool = True,
    phase_callback: Callable[[str, JsonDict], None] | None = None,
    *,
    log_console: Callable[[str], None] = _default_info_log,
    debug_log: Callable[[str], None] = _default_debug_log,
) -> dict[str, list[JsonDict]]:
    """KR: 로드된 UnityPy env에서 TTF/SDF 폰트 정보를 추출합니다.
    EN: Extracts TTF/SDF font information from a loaded UnityPy env.
    """
    scanned: dict[str, list[JsonDict]] = {"ttf": [], "sdf": []}
    texture_lookup: dict[tuple[str, int], Any] = {}
    texture_swizzle_cache: dict[str, str | None] = {}
    objects = env.objects
    try:
        object_count = len(objects)
    except Exception:
        object_count = None
    _emit_phase_callback(
        phase_callback,
        "object_scan_begin",
        file=file_name,
        object_count=object_count,
    )

    if scan_sdf and detect_ps5_swizzle:
        for item_index, item in enumerate(objects):
            if item_index % 256 == 0:
                _emit_phase_callback(
                    phase_callback,
                    "texture_index_progress",
                    file=file_name,
                    object_index=item_index,
                    object_count=object_count,
                )
            if item.type.name != "Texture2D":
                continue
            texture_lookup[(item.assets_file.name, int(item.path_id))] = item

    for object_index, obj in enumerate(objects):
        if object_index % 256 == 0:
            _emit_phase_callback(
                phase_callback,
                "object_scan_progress",
                file=file_name,
                object_index=object_index,
                object_count=object_count,
            )
        try:
            if scan_ttf and obj.type.name == "Font":
                font_name = obj.peek_name()
                if not font_name:
                    try:
                        font = obj.parse_as_object()
                        font_name = getattr(font, "m_Name", "") or ""
                    except Exception:
                        font_name = ""
                scanned["ttf"].append(
                    {
                        "file": file_name,
                        "assets_name": obj.assets_file.name,
                        "name": font_name,
                        "path_id": obj.path_id,
                    }
                )
            elif scan_sdf and obj.type.name == "MonoBehaviour":
                parse_dict = None
                atlas_file_id = 0
                atlas_path_id = 0
                glyph_count = 0
                try:
                    parse_dict = obj.parse_as_dict()
                    unity_version_hint = getattr(obj.assets_file, "unity_version", None)
                    tmp_info = inspect_tmp_font_schema(
                        parse_dict,
                        unity_version=(
                            str(unity_version_hint) if unity_version_hint else None
                        ),
                    )
                except Exception:
                    if lang == "ko":
                        debug_log(
                            f"[scan_fonts] parse_as_dict 실패: {file_name} | PathID {obj.path_id}"
                        )
                    else:
                        debug_log(
                            f"[scan_fonts] parse_as_dict failed: {file_name} | PathID {obj.path_id}"
                        )
                    continue

                if not tmp_info.get("is_tmp"):
                    continue

                try:
                    if parse_dict is None:
                        parse_dict = obj.parse_as_dict()
                    glyph_count = int(tmp_info.get("glyph_count", 0) or 0)
                    atlas_file_id = int(tmp_info.get("atlas_file_id", 0) or 0)
                    atlas_path_id = int(tmp_info.get("atlas_path_id", 0) or 0)
                    # KR: 외부 참조 stub(FileID!=0, PathID=0)은 실제 교체 대상이 아닙니다.
                    # EN: External reference stubs (FileID!=0, PathID=0) are not actual replacement targets.
                    if atlas_file_id != 0 and atlas_path_id == 0:
                        continue
                    if glyph_count == 0:
                        is_sprite_asset = (
                            parse_dict.get("spriteSheet") is not None
                            or isinstance(
                                parse_dict.get("m_SpriteCharacterTable"), list
                            )
                            or isinstance(parse_dict.get("m_SpriteGlyphTable"), list)
                            or isinstance(parse_dict.get("spriteInfoList"), list)
                        )
                        if is_sprite_asset:
                            continue
                        if atlas_file_id == 0 and atlas_path_id == 0:
                            continue
                except Exception:
                    if lang == "ko":
                        debug_log(
                            f"[scan_fonts] SDF 필드 검사 실패: {file_name} | PathID {obj.path_id}"
                        )
                    else:
                        debug_log(
                            f"[scan_fonts] SDF field check failed: {file_name} | PathID {obj.path_id}"
                        )
                    continue

                sdf_info: JsonDict = {
                    "file": file_name,
                    "assets_name": obj.assets_file.name,
                    "name": obj.peek_name(),
                    "path_id": obj.path_id,
                }
                if detect_ps5_swizzle:
                    swizzle_state = False
                    if atlas_file_id == 0 and atlas_path_id != 0:
                        cache_key = f"{obj.assets_file.name}|{atlas_path_id}"
                        if cache_key in texture_swizzle_cache:
                            swizzle_verdict = texture_swizzle_cache[cache_key]
                        else:
                            texture_obj = texture_lookup.get(
                                (obj.assets_file.name, atlas_path_id)
                            )
                            swizzle_verdict = (
                                detect_texture_object_ps5_swizzle(texture_obj)
                                if texture_obj is not None
                                else None
                            )
                            texture_swizzle_cache[cache_key] = swizzle_verdict
                        swizzle_state = swizzle_verdict == "likely_swizzled_input"
                    sdf_info["swizzle"] = "True" if swizzle_state else "False"

                scanned["sdf"].append(sdf_info)
        except Exception as e:
            if lang == "ko":
                log_console(
                    f"[scan_fonts] 오브젝트 처리 실패: {file_name} | PathID {obj.path_id} ({e})"
                )
            else:
                log_console(
                    f"[scan_fonts] Object processing failed: {file_name} | PathID {obj.path_id} ({e})"
                )
            continue

    _emit_phase_callback(
        phase_callback,
        "object_scan_end",
        file=file_name,
        object_count=object_count,
        ttf_count=len(scanned["ttf"]),
        sdf_count=len(scanned["sdf"]),
    )
    return scanned


def scan_fonts_in_asset_file(
    assets_file: str,
    generator: "TypeTreeGenerator | None",
    lang: Language = "ko",
    detect_ps5_swizzle: bool = False,
    scan_ttf: bool = True,
    scan_sdf: bool = True,
    phase_callback: Callable[[str, JsonDict], None] | None = None,
    *,
    load_environment: Callable[[str], Any] = load_unitypy,
    close_environment: Callable[[Any], None] = close_unitypy_env,
    scan_environment: Callable[..., dict[str, list[JsonDict]]] = scan_fonts_from_env,
) -> tuple[dict[str, list[JsonDict]], str | None]:
    """KR: 단일 에셋 파일을 로드해 폰트 정보를 추출합니다.
    EN: Loads a single asset file and extracts font information.
    """
    file_name = os.path.basename(assets_file)
    scanned: dict[str, list[JsonDict]] = {"ttf": [], "sdf": []}

    env = None
    _emit_phase_callback(
        phase_callback,
        "load_begin",
        file=file_name,
        path=assets_file,
    )
    try:
        env = load_environment(assets_file)
        env.typetree_generator = generator
    except Exception as e:
        _emit_phase_callback(
            phase_callback,
            "load_error",
            file=file_name,
            path=assets_file,
        )
        if lang == "ko":
            return scanned, f"UnityPy.load 실패: {assets_file} ({e})"
        return scanned, f"UnityPy.load failed: {assets_file} ({e})"

    _emit_phase_callback(
        phase_callback,
        "load_end",
        file=file_name,
        path=assets_file,
    )
    try:
        scanned = scan_environment(
            env,
            file_name,
            lang=lang,
            detect_ps5_swizzle=detect_ps5_swizzle,
            scan_ttf=scan_ttf,
            scan_sdf=scan_sdf,
            phase_callback=phase_callback,
        )
    finally:
        close_environment(env)
        env = None
        gc.collect()

    return scanned, None
