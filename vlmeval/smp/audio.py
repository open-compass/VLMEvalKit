import base64
import hashlib
import io
import json
import mimetypes
import os
import os.path as osp
import re
import shutil
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from .file import LMUDataRoot
from .log import get_logger

logger = get_logger(__name__)


class FileURIError(ValueError):
    """Raised when a file URI cannot be resolved to a local path."""


def file_uri_to_path(value):
    """Convert a local ``file://`` URI to a decoded filesystem path."""
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value.lower().startswith('file://'):
        raise FileURIError(f'Expected a file URI, got {value!r}.')

    parsed = urlparse(value)
    if parsed.netloc.lower() not in ('', 'localhost'):
        raise FileURIError(f'Unsupported non-local file URI: {value!r}.')
    return unquote(parsed.path)


@dataclass(frozen=True)
class AudioFormatSpec:
    """Canonical metadata for one supported audio format."""

    extension: str
    mime_type: str
    extension_aliases: tuple = ()
    container: str = None
    codec: str = None


# This is the single source of truth for extensions and preferred MIME types.
# OGG is a container, while Opus is a codec commonly carried in that container:
# generic ``audio/ogg`` therefore resolves to ``ogg``, while ``audio/opus`` and
# the dedicated .opus extension resolve to ``opus``. Provider-facing Opus bytes
# retain ``audio/ogg`` for backwards compatibility with existing integrations.
AUDIO_FORMAT_REGISTRY = {
    'wav': AudioFormatSpec('.wav', 'audio/wav', container='wav'),
    'mp3': AudioFormatSpec('.mp3', 'audio/mpeg', container='mp3', codec='mp3'),
    'm4a': AudioFormatSpec('.m4a', 'audio/mp4', container='mp4', codec='aac'),
    'flac': AudioFormatSpec('.flac', 'audio/flac', container='flac', codec='flac'),
    'ogg': AudioFormatSpec(
        '.ogg', 'audio/ogg', extension_aliases=('.oga',), container='ogg'
    ),
    'opus': AudioFormatSpec('.opus', 'audio/ogg', container='ogg', codec='opus'),
    'aac': AudioFormatSpec('.aac', 'audio/aac', container='adts', codec='aac'),
}

# MIME spellings which are not the preferred MIME declared in the registry.
# Values deliberately name a format rather than another MIME so audio/opus can
# distinguish the codec-specific declaration from generic audio/ogg.
AUDIO_MIME_ALIASES = {
    'audio/wave': 'wav',
    'audio/x-wav': 'wav',
    'audio/vnd.wave': 'wav',
    'audio/mp3': 'mp3',
    'audio/x-flac': 'flac',
    'audio/x-m4a': 'm4a',
    'audio/opus': 'opus',
    'audio/x-aac': 'aac',
}


def _derive_audio_maps():
    extension_to_format = {}
    extension_to_mime = {}
    format_to_mime = {}
    format_to_extension = {}
    mime_to_format = {}

    for audio_format, spec in AUDIO_FORMAT_REGISTRY.items():
        extensions = (spec.extension,) + spec.extension_aliases
        for extension in extensions:
            extension_to_format[extension] = audio_format
            extension_to_mime[extension] = spec.mime_type
        format_to_mime[audio_format] = spec.mime_type
        format_to_extension[audio_format] = spec.extension
        # audio/ogg is intentionally owned by the generic OGG entry; the later
        # Opus registry entry must not make a container MIME codec-specific.
        mime_to_format.setdefault(spec.mime_type, audio_format)

    mime_to_format.update(AUDIO_MIME_ALIASES)
    return (
        extension_to_mime,
        extension_to_format,
        mime_to_format,
        format_to_mime,
        format_to_extension,
    )


(
    AUDIO_MIME_MAP,
    AUDIO_FORMAT_MAP,
    AUDIO_MIME_FORMAT_MAP,
    AUDIO_FORMAT_MIME_MAP,
    AUDIO_FORMAT_EXTENSION_MAP,
) = _derive_audio_maps()

AUDIO_MEDIA_URL_PREFIXES = ('http://', 'https://', 'file://', 'data:audio/')
AUDIO_CACHE_PREFIX = 'vlmeval-audio-wav-v2-'
AUDIO_IO_CHUNK_SIZE = 1024 * 1024
_AUDIO_REUSE_HEADER_SIZE = 4096
_AUDIO_REUSE_SCAN_LIMIT = 1024 * 1024


@dataclass(frozen=True)
class AudioMetadata:
    """Metadata read from the actual audio stream rather than its file name."""

    sample_rate: int
    channels: int
    channel_layout: str
    frames: int = None
    duration: float = None


@dataclass(frozen=True)
class AudioPayload:
    """Validated audio bytes and the format that a provider should receive."""

    data: bytes
    mime_type: str
    format: str
    source: str
    source_metadata: AudioMetadata = field(default=None, compare=False, repr=False)
    sent_metadata: AudioMetadata = field(default=None, compare=False, repr=False)

    @property
    def sample_rate(self):
        metadata = self.sent_metadata or self.source_metadata
        return metadata.sample_rate if metadata is not None else None

    @property
    def channels(self):
        metadata = self.sent_metadata or self.source_metadata
        return metadata.channels if metadata is not None else None

    @property
    def channel_layout(self):
        metadata = self.sent_metadata or self.source_metadata
        return metadata.channel_layout if metadata is not None else None


class AudioNormalizationError(ValueError):
    """Raised when an audio source cannot be validated or normalized."""


class AudioDownloadError(AudioNormalizationError):
    """Raised for retryable failures while localizing a remote audio source."""


class AudioSizeLimitError(AudioNormalizationError):
    """Raised when an audio input or normalized payload exceeds its policy."""


class UnsupportedAudioSourceError(AudioDownloadError):
    """Raised when a media value cannot be identified as a supported audio source."""


def validate_audio_size_limit(max_file_size):
    if max_file_size is None:
        return None
    if isinstance(max_file_size, bool) or not isinstance(max_file_size, (int, float)):
        raise AudioNormalizationError('max_file_size must be a positive integer or None.')
    if isinstance(max_file_size, float) and not max_file_size.is_integer():
        raise AudioNormalizationError('max_file_size must be a positive integer or None.')
    try:
        max_file_size = int(max_file_size)
    except (OverflowError, ValueError) as err:
        raise AudioNormalizationError(
            'max_file_size must be a positive integer or None.'
        ) from err
    if max_file_size <= 0:
        raise AudioNormalizationError('max_file_size must be a positive integer or None.')
    return max_file_size


def check_audio_size(size, max_file_size, source):
    max_file_size = validate_audio_size_limit(max_file_size)
    if max_file_size is not None and size > max_file_size:
        raise AudioSizeLimitError(
            f'Audio source {source!r} is {size} bytes, exceeding the allowed '
            f'maximum of {max_file_size} bytes.'
        )


class _BoundedAudioBuffer(io.BytesIO):
    """Seekable buffer that rejects writes before they grow past a byte limit."""

    def __init__(self, max_file_size, source):
        super().__init__()
        self._max_file_size = validate_audio_size_limit(max_file_size)
        self._source = source
        self._extent = 0

    def write(self, data):
        proposed_extent = max(self._extent, self.tell() + len(data))
        check_audio_size(proposed_extent, self._max_file_size, source=self._source)
        written = super().write(data)
        self._extent = max(self._extent, self.tell())
        return written

    def truncate(self, size=None):
        target_size = self.tell() if size is None else size
        check_audio_size(target_size, self._max_file_size, source=self._source)
        result = super().truncate(size)
        self._extent = result
        return result


def base64_decoded_size(encoded):
    compact = re.sub(r'\s+', '', encoded)
    padding = len(compact) - len(compact.rstrip('='))
    return compact, (len(compact) + 3) // 4 * 3 - min(padding, 2)


def atomic_write_audio_file(path, writer):
    """Atomically publish writer output only after common audio validation."""
    path = os.fspath(path)
    parent = osp.dirname(osp.abspath(path))
    os.makedirs(parent, exist_ok=True)
    suffix = osp.splitext(path)[1] or '.audio'
    fd, tmp_path = tempfile.mkstemp(
        prefix=f'.{osp.basename(path)}.', suffix=suffix, dir=parent
    )
    try:
        with os.fdopen(fd, 'wb') as output:
            fd = None
            writer(output)
            output.flush()
            os.fsync(output.fileno())
        try:
            _validate_audio_file_for_publication(tmp_path)
        except AudioNormalizationError as err:
            raise AudioNormalizationError(
                f'Refusing to publish invalid audio file: {path!r}.'
            ) from err
        os.replace(tmp_path, path)
    finally:
        if fd is not None:
            os.close(fd)
        if osp.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return path


def suffix_from_media_value(value):
    value = str(value)
    value = value.split('?', 1)[0].split('#', 1)[0]
    return osp.splitext(value)[1].lower()


_suffix_from_media_value = suffix_from_media_value


def infer_audio_mime_type(value):
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str):
        return None
    if value.lower().startswith('data:audio/'):
        mime = value[5:].split(';', 1)[0].lower()
        return mime if mime.startswith('audio/') else None

    suffix = suffix_from_media_value(value)
    if suffix in AUDIO_MIME_MAP:
        return AUDIO_MIME_MAP[suffix]
    mime = mimetypes.guess_type(value)[0]
    if mime and mime.startswith('audio/'):
        return mime
    return None


def audio_mime_type(value, default='audio/wav'):
    mime = infer_audio_mime_type(value)
    if mime is not None:
        return mime
    if default is not None:
        return default
    raise AudioNormalizationError(f'Unable to determine audio MIME type for {value!r}.')


def is_audio_media_url(value):
    return isinstance(value, str) and value.lower().startswith(AUDIO_MEDIA_URL_PREFIXES)


def _remote_content_type(url):
    try:
        request = Request(url, method='HEAD', headers={'User-Agent': 'VLMEvalKit'})
        with urlopen(request, timeout=30) as response:
            content_type = response.headers.get('Content-Type')
    except Exception as err:
        logger.debug(f'Unable to inspect Content-Type for {url!r}: {err}')
        return None
    if content_type is None:
        return None
    return content_type.split(';', 1)[0].strip().lower()


def _download_audio_file(url, filename, max_file_size=None):
    """Download and validate audio before atomically publishing it."""
    max_file_size = validate_audio_size_limit(max_file_size)
    parent = osp.dirname(osp.abspath(filename))
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        prefix=f'.{osp.basename(filename)}.', suffix='.download', dir=parent, delete=False
    )
    tmp_path = tmp.name
    tmp.close()
    try:
        request = Request(url, headers={'User-Agent': 'VLMEvalKit'})
        with urlopen(request, timeout=60) as response, open(tmp_path, 'wb') as output:
            content_length = response.headers.get('Content-Length')
            expected_size = None
            if content_length is not None:
                try:
                    expected_size = int(content_length)
                except (TypeError, ValueError):
                    logger.warning(
                        f'Ignoring invalid Content-Length for {url!r}: {content_length!r}'
                    )
                else:
                    if expected_size < 0:
                        logger.warning(
                            f'Ignoring invalid Content-Length for {url!r}: {content_length!r}'
                        )
                        expected_size = None
                    else:
                        check_audio_size(expected_size, max_file_size, source=url)
            total = 0
            while True:
                chunk = response.read(AUDIO_IO_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                check_audio_size(total, max_file_size, source=url)
                output.write(chunk)
            if expected_size is not None and total != expected_size:
                raise AudioDownloadError(
                    f'Incomplete audio download from {url!r}: expected '
                    f'{expected_size} bytes, received {total} bytes.'
                )
            output.flush()
            os.fsync(output.fileno())

        with open(tmp_path, 'rb') as downloaded:
            detected_format = detect_audio_format(downloaded.read(4096))
        if detected_format is None:
            raise AudioNormalizationError(
                f'Unknown or unsupported audio container downloaded from {url!r}.'
            )
        validated_tmp_path = tmp_path + AUDIO_FORMAT_EXTENSION_MAP[detected_format]
        os.replace(tmp_path, validated_tmp_path)
        tmp_path = validated_tmp_path
        _validate_audio_file_for_publication(tmp_path)
        os.replace(tmp_path, filename)
    except AudioSizeLimitError:
        raise
    except Exception as err:
        logger.warning(f'{type(err)}: {err}')
        if 'huggingface.co' in url:
            mirror_url = url.replace('huggingface.co', 'hf-mirror.com')
            try:
                return _download_audio_file(
                    mirror_url, filename, max_file_size=max_file_size
                )
            except AudioNormalizationError:
                raise
            except Exception as mirror_err:
                logger.warning(f'{type(mirror_err)}: {mirror_err}')
                raise AudioDownloadError(f'Failed to download {url}') from mirror_err
        if isinstance(err, AudioNormalizationError):
            raise
        raise AudioDownloadError(f'Failed to download {url}') from err
    finally:
        if osp.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return filename


def _validated_cached_audio_format(path, max_file_size=None, source=None):
    """Validate a localized cache entry and return its detected format."""
    source = path if source is None else source
    check_audio_size(osp.getsize(path), max_file_size, source=source)
    with open(path, 'rb') as stream:
        detected_format = detect_audio_format(stream.read(4096))
    if detected_format is None:
        raise AudioNormalizationError(
            f'Unknown or unsupported audio container for source {source!r}.'
        )
    _validate_audio_file_for_publication(path)
    return detected_format


def _resolve_audio_url(url, mime_type, max_file_size=None):
    declared_format = AUDIO_FORMAT_MAP.get(suffix_from_media_value(url))
    if declared_format is None:
        declared_format = AUDIO_MIME_FORMAT_MAP.get(mime_type)
    if declared_format is None:
        raise AudioNormalizationError(
            f'Unsupported HTTP audio Content-Type: {mime_type!r}.'
        )
    cache_dir = osp.join(LMUDataRoot(), 'files')
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5(url.encode('utf-8')).hexdigest()
    target = osp.join(cache_dir, key + AUDIO_FORMAT_EXTENSION_MAP[declared_format])

    candidate_formats = [declared_format]
    candidate_formats.extend(
        audio_format
        for audio_format in AUDIO_FORMAT_EXTENSION_MAP
        if audio_format != declared_format
    )
    for candidate_format in candidate_formats:
        candidate = osp.join(
            cache_dir, key + AUDIO_FORMAT_EXTENSION_MAP[candidate_format]
        )
        if not osp.isfile(candidate):
            continue
        try:
            detected_format = _validated_cached_audio_format(
                candidate, max_file_size=max_file_size, source=url
            )
            final_candidate = osp.join(
                cache_dir, key + AUDIO_FORMAT_EXTENSION_MAP[detected_format]
            )
            if final_candidate != candidate:
                os.replace(candidate, final_candidate)
            return final_candidate
        except (AudioNormalizationError, OSError):
            logger.warning(f'Ignoring invalid localized audio file: {candidate}')

    _download_audio_file(url, target, max_file_size=max_file_size)
    detected_format = _validated_cached_audio_format(
        target, max_file_size=max_file_size, source=url
    )
    final_target = osp.join(
        cache_dir, key + AUDIO_FORMAT_EXTENSION_MAP[detected_format]
    )
    if final_target != target:
        os.replace(target, final_target)
    return final_target


def _decode_audio_data_uri(value, max_file_size=None):
    try:
        header, encoded = value.split(',', 1)
    except ValueError as err:
        raise AudioNormalizationError('Malformed audio data URI: missing comma.') from err

    header_parts = header[5:].split(';')
    mime_type = header_parts[0].lower()
    params = {part.lower() for part in header_parts[1:]}
    declared_format = AUDIO_MIME_FORMAT_MAP.get(mime_type)
    if declared_format is None:
        raise AudioNormalizationError(
            f'Unsupported audio MIME type in data URI: {mime_type!r}.'
        )
    if 'base64' not in params:
        raise AudioNormalizationError('Audio data URI must use base64 encoding.')
    try:
        compact, decoded_size = base64_decoded_size(encoded)
        check_audio_size(decoded_size, max_file_size, source=f'data:{mime_type};base64')
        data = base64.b64decode(compact, validate=True)
    except Exception as err:
        if isinstance(err, AudioNormalizationError):
            raise
        raise AudioNormalizationError('Audio data URI contains invalid base64 data.') from err

    detected_format = detect_audio_format(data)
    if detected_format is None:
        raise AudioNormalizationError(
            f'Unknown or unsupported audio container for data:{mime_type};base64.'
        )
    if not _compatible_audio_formats(declared_format, detected_format):
        raise AudioNormalizationError(
            f'Audio format mismatch for data:{mime_type};base64: declared '
            f'{declared_format}, content is {detected_format}.'
        )
    return data, detected_format


def _resolve_audio_data_uri(value, max_file_size=None):
    data, detected_format = _decode_audio_data_uri(
        value, max_file_size=max_file_size
    )
    cache_dir = osp.join(LMUDataRoot(), 'files')
    key = hashlib.sha256(data).hexdigest()
    target = osp.join(
        cache_dir, key + AUDIO_FORMAT_EXTENSION_MAP[detected_format]
    )
    if osp.isfile(target):
        try:
            cached_format = _validated_cached_audio_format(
                target,
                max_file_size=max_file_size,
                source=f'data:{AUDIO_FORMAT_MIME_MAP[detected_format]};base64',
            )
            if cached_format == detected_format:
                return target
        except (AudioNormalizationError, OSError):
            logger.warning(f'Ignoring invalid localized audio file: {target}')

    atomic_write_audio_file(target, lambda output: output.write(data))
    return target


def resolve_media_source(value, max_file_size=None):
    """Resolve one audio media value to the local-path intermediate protocol.

    Local paths are checked and returned directly. Local ``file://`` URIs are
    decoded, data URIs are atomically cached by content, and HTTP(S) sources are
    downloaded with a byte limit before an atomically published cache entry is
    returned. Provider-specific normalization intentionally happens later.
    """
    max_file_size = validate_audio_size_limit(max_file_size)
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value:
        raise AudioNormalizationError('Audio source must be a non-empty string or path.')

    lower_value = value.lower()
    if lower_value.startswith('data:audio/'):
        return _resolve_audio_data_uri(value, max_file_size=max_file_size)

    if lower_value.startswith('file://'):
        try:
            path = file_uri_to_path(value)
        except FileURIError as err:
            raise AudioNormalizationError(str(err)) from err
    elif lower_value.startswith(('http://', 'https://')):
        mime_type = infer_audio_mime_type(value) or _remote_content_type(value)
        if mime_type is None or not mime_type.startswith('audio/'):
            raise UnsupportedAudioSourceError(
                f'Unable to identify remote audio source {value!r} from its URL '
                'or Content-Type.'
            )
        return _resolve_audio_url(value, mime_type, max_file_size=max_file_size)
    else:
        path = value

    if not osp.isfile(path):
        raise AudioNormalizationError(f'Audio file does not exist: {path!r}.')
    suffix = suffix_from_media_value(path)
    if suffix not in AUDIO_FORMAT_MAP:
        raise AudioNormalizationError(
            f'Unsupported or missing audio file extension: {suffix or "<none>"}.'
        )
    check_audio_size(osp.getsize(path), max_file_size, source=value)
    return path


def detect_audio_format(data):
    """Return a format from file signatures, without trusting a name or MIME."""
    if len(data) >= 12 and data[:4] in (b'RIFF', b'RF64') and data[8:12] == b'WAVE':
        return 'wav'
    if data.startswith(b'fLaC'):
        return 'flac'
    if data.startswith(b'OggS'):
        return 'opus' if b'OpusHead' in data[:4096] else 'ogg'
    if len(data) >= 12 and data[4:8] == b'ftyp':
        return 'm4a'
    if data.startswith(b'ID3'):
        return 'mp3'
    if len(data) >= 2 and data[0] == 0xFF:
        # ADTS has a 12-bit sync word and its two layer bits are always zero.
        if data[1] & 0xF6 == 0xF0:
            return 'aac'
        # MPEG audio has a valid version and a non-zero layer field.
        if data[1] & 0xE0 == 0xE0 and data[1] & 0x18 != 0x08 and data[1] & 0x06:
            return 'mp3'
    return None


_detect_audio_format = detect_audio_format


def _read_audio_range(path, offset, size):
    if offset < 0 or size < 0:
        return b''
    with open(path, 'rb') as stream:
        stream.seek(offset)
        return stream.read(size)


def _audio_bytes_at(path, prefix, offset, size):
    if offset < len(prefix) and offset + size <= len(prefix):
        return prefix[offset:offset + size]
    return _read_audio_range(path, offset, size)


def _flac_crc8(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


@lru_cache(maxsize=1)
def _flac_crc16_table():
    table = []
    for byte in range(256):
        crc = byte << 8
        for _ in range(8):
            crc = (
                ((crc << 1) ^ 0x8005) & 0xFFFF
                if crc & 0x8000 else (crc << 1) & 0xFFFF
            )
        table.append(crc)
    return tuple(table)


def _valid_flac_frame_header(data):
    if len(data) < 6 or data[0] != 0xFF or data[1] & 0xFC != 0xF8:
        return False
    if data[1] & 0x02 or data[3] & 0x01:
        return False
    block_size_code = data[2] >> 4
    sample_rate_code = data[2] & 0x0F
    channel_assignment = data[3] >> 4
    sample_size_code = (data[3] >> 1) & 0x07
    if (
        block_size_code == 0
        or sample_rate_code == 15
        or channel_assignment > 10
        or sample_size_code in (3, 7)
    ):
        return False

    number_start = 4
    first = data[number_start]
    if first & 0x80 == 0:
        number_length = 1
    elif first & 0xE0 == 0xC0:
        number_length = 2
    elif first & 0xF0 == 0xE0:
        number_length = 3
    elif first & 0xF8 == 0xF0:
        number_length = 4
    elif first & 0xFC == 0xF8:
        number_length = 5
    elif first & 0xFE == 0xFC:
        number_length = 6
    elif first == 0xFE:
        number_length = 7
    else:
        return False
    index = number_start + number_length
    if index >= len(data):
        return False
    if any(byte & 0xC0 != 0x80 for byte in data[number_start + 1:index]):
        return False

    if block_size_code == 6:
        index += 1
    elif block_size_code == 7:
        index += 2
    if sample_rate_code == 12:
        index += 1
    elif sample_rate_code in (13, 14):
        index += 2
    if index >= len(data):
        return False
    return _flac_crc8(data[:index + 1]) == 0


def _flac_frame_crc_ok(path, frame_start, file_size):
    table = _flac_crc16_table()
    crc = 0
    remaining = file_size - frame_start
    with open(path, 'rb') as stream:
        stream.seek(frame_start)
        while remaining:
            chunk = stream.read(min(AUDIO_IO_CHUNK_SIZE, remaining))
            if not chunk:
                return False
            remaining -= len(chunk)
            for byte in chunk:
                crc = ((crc << 8) & 0xFFFF) ^ table[((crc >> 8) ^ byte) & 0xFF]
    return crc == 0


def _reusable_riff_wave(path, prefix, file_size):
    header = _audio_bytes_at(path, prefix, 0, 12)
    if len(header) != 12 or header[:4] not in (b'RIFF', b'RF64') or header[8:] != b'WAVE':
        return False

    is_rf64 = header[:4] == b'RF64'
    riff_end = file_size if is_rf64 else int.from_bytes(header[4:8], 'little') + 8
    if riff_end < 12 or riff_end > file_size:
        return False
    offset = 12
    rf64_data_size = None
    found_fmt = False
    block_align = 0
    data_size = 0
    for _ in range(4096):
        chunk_header = _audio_bytes_at(path, prefix, offset, 8)
        if len(chunk_header) != 8:
            return False
        chunk_type = chunk_header[:4]
        chunk_size = int.from_bytes(chunk_header[4:], 'little')
        payload_offset = offset + 8

        if is_rf64 and chunk_type == b'ds64':
            if offset != 12 or chunk_size < 28:
                return False
            ds64 = _audio_bytes_at(path, prefix, payload_offset, 28)
            if len(ds64) != 28:
                return False
            riff_end = int.from_bytes(ds64[:8], 'little') + 8
            rf64_data_size = int.from_bytes(ds64[8:16], 'little')
            if riff_end < payload_offset + chunk_size or riff_end > file_size:
                return False

        effective_size = chunk_size
        if chunk_type == b'data' and chunk_size == 0xFFFFFFFF:
            if not is_rf64 or rf64_data_size is None:
                return False
            effective_size = rf64_data_size
        elif chunk_size == 0xFFFFFFFF:
            return False
        payload_end = payload_offset + effective_size
        padded_end = payload_end + (effective_size & 1)
        if payload_end > riff_end or padded_end > riff_end:
            return False

        if chunk_type == b'fmt ':
            if chunk_size < 16:
                return False
            fmt = _audio_bytes_at(path, prefix, payload_offset, 16)
            if len(fmt) != 16:
                return False
            audio_format = int.from_bytes(fmt[:2], 'little')
            channels = int.from_bytes(fmt[2:4], 'little')
            sample_rate = int.from_bytes(fmt[4:8], 'little')
            byte_rate = int.from_bytes(fmt[8:12], 'little')
            block_align = int.from_bytes(fmt[12:14], 'little')
            if not all((audio_format, channels, sample_rate, byte_rate, block_align)):
                return False
            found_fmt = True
        elif chunk_type == b'data':
            data_size = effective_size
        if found_fmt and data_size:
            return data_size >= block_align
        offset = padded_end
        if offset + 8 > riff_end:
            break
    return False


def _mp3_frame_length(header):
    if len(header) < 4 or header[0] != 0xFF or header[1] & 0xE0 != 0xE0:
        return None
    version = (header[1] >> 3) & 0x03
    layer = (header[1] >> 1) & 0x03
    bitrate_index = (header[2] >> 4) & 0x0F
    sample_rate_index = (header[2] >> 2) & 0x03
    padding = (header[2] >> 1) & 0x01
    if version == 1 or layer == 0 or bitrate_index in (0, 15) or sample_rate_index == 3:
        return None

    bitrate_tables = {
        3: {
            3: (32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448),
            2: (32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384),
            1: (32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
        },
        2: {
            3: (32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256),
            2: (8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
            1: (8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
        },
        0: {
            3: (32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256),
            2: (8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
            1: (8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
        },
    }
    bitrate = bitrate_tables[version][layer][bitrate_index - 1] * 1000
    sample_rate = (44100, 48000, 32000)[sample_rate_index]
    if version == 2:
        sample_rate //= 2
    elif version == 0:
        sample_rate //= 4
    if layer == 3:
        return (12 * bitrate // sample_rate + padding) * 4
    if layer == 1 and version != 3:
        return 72 * bitrate // sample_rate + padding
    return 144 * bitrate // sample_rate + padding


def _reusable_mp3(path, prefix, file_size):
    offset = 0
    if prefix[:3] == b'ID3' or _audio_bytes_at(path, prefix, 0, 3) == b'ID3':
        header = _audio_bytes_at(path, prefix, 0, 10)
        if len(header) < 10 or header[3] not in (2, 3, 4):
            return False
        if any(byte & 0x80 for byte in header[6:10]):
            return False
        tag_size = sum((header[index] & 0x7F) << (7 * (9 - index)) for index in range(6, 10))
        offset = 10 + tag_size + (10 if header[5] & 0x10 else 0)
        if offset > file_size:
            return False

    frame_count = 0
    while offset + 4 <= file_size and frame_count < 100000:
        header = _audio_bytes_at(path, prefix, offset, 4)
        frame_length = _mp3_frame_length(header)
        if frame_length is None:
            if frame_count == 0:
                offset += 1
                continue
            # ID3v1/APEv2 trailers are legal after the last frame. Avoid
            # accepting arbitrary padding as a complete MP3 stream.
            remaining = file_size - offset
            trailer = _audio_bytes_at(path, prefix, offset, min(128, remaining))
            if trailer.startswith(b'TAG'):
                return remaining == 128
            if trailer.startswith(b'APETAGEX'):
                if len(trailer) < 32:
                    return False
                tag_size = int.from_bytes(trailer[12:16], 'little')
                return tag_size >= 32 and remaining in (tag_size, tag_size + 32)
            return False
        if frame_length < 4 or offset + frame_length > file_size:
            return False
        frame_count += 1
        offset += frame_length
    return frame_count > 0 and offset == file_size


def _reusable_adts(path, prefix, file_size):
    offset = 0
    frame_count = 0
    while offset + 7 <= file_size and frame_count < 100000:
        header = _audio_bytes_at(path, prefix, offset, 9)
        if len(header) < 7 or header[0] != 0xFF or header[1] & 0xF6 != 0xF0:
            return False
        sample_rate_index = (header[2] >> 2) & 0x0F
        channel_config = ((header[2] & 0x01) << 2) | ((header[3] >> 6) & 0x03)
        protection_absent = header[1] & 0x01
        header_length = 7 if protection_absent else 9
        frame_length = ((header[3] & 0x03) << 11) | (header[4] << 3) | (header[5] >> 5)
        if sample_rate_index >= 13 or channel_config == 0:
            return False
        if frame_length <= header_length or offset + frame_length > file_size:
            return False
        frame_count += 1
        offset += frame_length
    return frame_count > 0 and offset == file_size


def _reusable_flac(path, prefix, file_size):
    if file_size < 8 or _audio_bytes_at(path, prefix, 0, 4) != b'fLaC':
        return False
    offset = 4
    streaminfo = None
    for _ in range(128):
        block_header = _audio_bytes_at(path, prefix, offset, 4)
        if len(block_header) < 4:
            return False
        block_type = block_header[0] & 0x7F
        block_length = int.from_bytes(block_header[1:4], 'big')
        offset += 4
        if block_type == 127 or offset + block_length > file_size:
            return False
        if block_type == 0:
            if block_length != 34 or streaminfo is not None:
                return False
            streaminfo = _audio_bytes_at(path, prefix, offset, block_length)
            if len(streaminfo) != block_length:
                return False
            sample_rate = (streaminfo[10] << 12) | (streaminfo[11] << 4) | (streaminfo[12] >> 4)
            channels = ((streaminfo[12] & 0x0E) >> 1) + 1
            bits_per_sample = ((streaminfo[12] & 0x01) << 4) | (streaminfo[13] >> 4)
            bits_per_sample += 1
            if sample_rate <= 0 or channels > 8 or bits_per_sample < 4 or bits_per_sample > 32:
                return False
        offset += block_length
        if block_header[0] & 0x80:
            break
    else:
        return False
    if streaminfo is None or offset >= file_size:
        return False
    scan_start = max(offset, file_size - _AUDIO_REUSE_SCAN_LIMIT)
    frame_data = _read_audio_range(path, scan_start, file_size - scan_start)
    candidates = [
        scan_start + index
        for index in range(len(frame_data) - 1)
        if frame_data[index] == 0xFF and frame_data[index + 1] & 0xFC == 0xF8
    ]
    if not candidates or file_size < 2:
        return False

    # A FLAC frame ends with a CRC-16. Checking the final frame's checksum
    # catches the common case where a cached download was truncated after a
    # valid header while remaining independent of a decoder subprocess.
    valid_header_candidates = 0
    for frame_start in reversed(candidates):
        if frame_start + 4 >= file_size - 2:
            continue
        frame_header = _read_audio_range(path, frame_start, min(32, file_size - frame_start))
        if not _valid_flac_frame_header(frame_header):
            continue
        valid_header_candidates += 1
        if _flac_frame_crc_ok(path, frame_start, file_size):
            return True
        if valid_header_candidates >= 4:
            break
    return False


def _reusable_ogg(path, prefix, file_size):
    offset = 0
    pending = bytearray()
    packet_count = 0
    expected_headers = None
    audio_packet = False
    page_count = 0
    final_page_eos = False
    while offset < file_size and page_count < 4096:
        header = _audio_bytes_at(path, prefix, offset, 27)
        if len(header) < 27 or header[:4] != b'OggS' or header[4] != 0:
            return False
        continued = bool(header[5] & 0x01)
        if continued != bool(pending):
            return False
        segment_count = header[26]
        lacing = _audio_bytes_at(path, prefix, offset + 27, segment_count)
        if len(lacing) != segment_count:
            return False
        payload_size = sum(lacing)
        payload_offset = offset + 27 + segment_count
        payload = _read_audio_range(path, payload_offset, payload_size)
        if len(payload) != payload_size or payload_offset + payload_size > file_size:
            return False
        cursor = 0
        for segment_length in lacing:
            if len(pending) + segment_length > _AUDIO_REUSE_SCAN_LIMIT:
                return False
            pending.extend(payload[cursor:cursor + segment_length])
            cursor += segment_length
            if segment_length < 255:
                packet_count += 1
                if packet_count == 1:
                    if pending.startswith(b'OpusHead'):
                        expected_headers = 2
                    elif pending.startswith(b'\x01vorbis'):
                        expected_headers = 3
                    elif pending.startswith(b'Speex   '):
                        expected_headers = 1
                    else:
                        expected_headers = 1
                elif packet_count > expected_headers:
                    audio_packet = True
                pending.clear()
        offset = payload_offset + payload_size
        page_count += 1
        final_page_eos = bool(header[5] & 0x04)
    return (
        offset == file_size
        and not pending
        and packet_count > 0
        and audio_packet
        and final_page_eos
    )


def _reusable_mp4(path, prefix, file_size):
    if file_size < 16 or _audio_bytes_at(path, prefix, 4, 4) != b'ftyp':
        return False
    offset = 0
    box_types = []
    while offset < file_size and len(box_types) < 4096:
        header = _audio_bytes_at(path, prefix, offset, 8)
        if len(header) < 8:
            return False
        size32 = int.from_bytes(header[:4], 'big')
        box_type = header[4:8]
        header_size = 8
        if size32 == 1:
            extended = _audio_bytes_at(path, prefix, offset + 8, 8)
            if len(extended) < 8:
                return False
            box_size = int.from_bytes(extended, 'big')
            header_size = 16
        elif size32 == 0:
            box_size = file_size - offset
        else:
            box_size = size32
        if box_size < header_size or offset + box_size > file_size:
            return False
        if offset == 0 and (box_type != b'ftyp' or box_size < 16):
            return False
        box_types.append(box_type)
        offset += box_size
    return offset == file_size and b'mdat' in box_types and bool(
        {b'moov', b'moof'} & set(box_types)
    )


def is_reusable_audio_file(path):
    """Cheaply check a repository cache entry without a decoder subprocess.

    Newly written files receive full decoder-backed validation before publication.
    Reuse scans therefore only verify the declared/detected container and, for PCM
    WAV, use the standard library parser to seek to and read the final frame.
    """
    try:
        path = os.fspath(path)
        with open(path, 'rb') as stream:
            file_size = os.fstat(stream.fileno()).st_size
            if file_size < 16:
                return False
            prefix = stream.read(min(_AUDIO_REUSE_HEADER_SIZE, file_size))
        detected_format = detect_audio_format(prefix)
        declared_format = AUDIO_FORMAT_MAP.get(suffix_from_media_value(path))
        # MMSU contains valid files whose suffix does not match their actual
        # container. Require a supported cache-file suffix, then validate the
        # detected container rather than treating the name as authoritative.
        if detected_format is None or declared_format is None:
            return False
        if detected_format != 'wav':
            validators = {
                'mp3': _reusable_mp3,
                'aac': _reusable_adts,
                'flac': _reusable_flac,
                'ogg': _reusable_ogg,
                'opus': _reusable_ogg,
                'm4a': _reusable_mp4,
            }
            validator = validators.get(detected_format)
            return validator is not None and validator(path, prefix, file_size)

        try:
            with wave.open(path, 'rb') as wav:
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                frames = wav.getnframes()
                if channels <= 0 or sample_width <= 0 or wav.getframerate() <= 0 or frames <= 0:
                    return False
                wav.setpos(frames - 1)
                return len(wav.readframes(1)) == channels * sample_width
        except (EOFError, ValueError, wave.Error):
            return _reusable_riff_wave(path, prefix, file_size)
    except (EOFError, OSError, TypeError, ValueError, wave.Error):
        return False


def _compatible_audio_formats(declared, detected):
    if declared == detected:
        return True
    # Opus commonly uses an .ogg name and audio/ogg MIME.
    return {declared, detected} == {'ogg', 'opus'}


def _positive_metadata_int(value, field, source):
    try:
        value = int(value)
    except (TypeError, ValueError) as err:
        raise AudioNormalizationError(
            f'Unable to detect a valid audio {field} for {source!r}.'
        ) from err
    if value <= 0:
        raise AudioNormalizationError(f'Unable to detect a valid audio {field} for {source!r}.')
    return value


def _validate_pcm_wav_frames(data, source):
    """Catch empty/truncated PCM WAV files without fully decoding compressed inputs."""
    try:
        with wave.open(io.BytesIO(data), 'rb') as wav:
            channels = _positive_metadata_int(wav.getnchannels(), 'channel count', source)
            sample_rate = _positive_metadata_int(wav.getframerate(), 'sample rate', source)
            frames = _positive_metadata_int(wav.getnframes(), 'frame count', source)
            sample_width = _positive_metadata_int(wav.getsampwidth(), 'sample width', source)
            frame_width = channels * sample_width
            wav.setpos(frames - 1)
            final_frame = wav.readframes(1)
            if len(final_frame) < frame_width:
                raise AudioNormalizationError(
                    f'Audio source {source!r} is a truncated WAV: expected at least '
                    f'{frames * frame_width} PCM bytes, final frame has '
                    f'{len(final_frame)} of {frame_width} bytes.'
                )
    except AudioNormalizationError:
        raise
    except (EOFError, wave.Error) as err:
        # Python's wave module does not support every valid WAVE codec or RF64.
        # The actual decoder probe below remains authoritative for those files.
        logger.debug(f'Built-in WAV validation deferred to decoder for {source!r}: {err}')
        return None
    return AudioMetadata(
        sample_rate=sample_rate,
        channels=channels,
        channel_layout='unknown',
        frames=frames,
        duration=frames / sample_rate,
    )


def _probe_audio_with_pyav(data, source):
    try:
        import av
    except ImportError:
        return None

    try:
        with av.open(io.BytesIO(data), mode='r') as container:
            streams = [stream for stream in container.streams if stream.type == 'audio']
            if not streams:
                raise AudioNormalizationError(f'No audio stream found in {source!r}.')
            stream = streams[0]
            codec = stream.codec_context
            sample_rate = _positive_metadata_int(
                codec.sample_rate or getattr(stream, 'rate', None), 'sample rate', source
            )
            channels = _positive_metadata_int(codec.channels, 'channel count', source)
            layout = getattr(codec.layout, 'name', None)
            if not layout:
                raise AudioNormalizationError(
                    f'Unable to detect an audio channel layout for {source!r}.'
                )

            first_frame = None
            for frame in container.decode(stream):
                if getattr(frame, 'samples', 0) > 0:
                    first_frame = frame
                    break
            if first_frame is None:
                raise AudioNormalizationError(
                    f'Audio stream in {source!r} contains no decodable frames.'
                )

            duration = None
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            frames = None
            if duration is not None and duration > 0:
                frames = max(1, int(round(duration * sample_rate)))
            elif getattr(first_frame, 'samples', 0):
                frames = int(first_frame.samples)
                duration = frames / sample_rate
            return AudioMetadata(
                sample_rate=sample_rate,
                channels=channels,
                channel_layout=str(layout),
                frames=frames,
                duration=duration,
            )
    except AudioNormalizationError:
        raise
    except Exception as err:
        raise AudioNormalizationError(
            f'Unable to decode audio metadata from {source!r} with PyAV: {err}'
        ) from err


def _probe_audio_with_ffprobe(data, source):
    ffprobe = shutil.which('ffprobe')
    if ffprobe is None:
        return None
    command = [
        ffprobe, '-v', 'error', '-select_streams', 'a:0',
        '-show_entries', 'stream=sample_rate,channels,channel_layout,duration,nb_frames',
        '-of', 'json', '-i', 'pipe:0',
    ]
    try:
        result = subprocess.run(
            command,
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as err:
        raise AudioNormalizationError(
            f'Unable to probe audio metadata from {source!r} with ffprobe: {err}'
        ) from err
    if result.returncode != 0:
        error = result.stderr.decode('utf-8', errors='replace').strip()
        raise AudioNormalizationError(
            f'Unable to decode audio metadata from {source!r} with ffprobe: {error}'
        )
    try:
        streams = json.loads(result.stdout.decode('utf-8')).get('streams', [])
        stream = streams[0]
        sample_rate = _positive_metadata_int(stream.get('sample_rate'), 'sample rate', source)
        channels = _positive_metadata_int(stream.get('channels'), 'channel count', source)
        layout = stream.get('channel_layout')
        if not layout:
            raise AudioNormalizationError(
                f'Unable to detect an audio channel layout for {source!r}.'
            )
        duration_raw = stream.get('duration')
        duration = float(duration_raw) if duration_raw not in (None, 'N/A') else None
        frames_raw = stream.get('nb_frames')
        frames = int(frames_raw) if frames_raw not in (None, 'N/A') else None
        return AudioMetadata(sample_rate, channels, layout, frames, duration)
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as err:
        raise AudioNormalizationError(
            f'ffprobe returned incomplete audio metadata for {source!r}: {err}'
        ) from err


def _probe_audio_metadata(data, source_format, source):
    wav_metadata = None
    if source_format == 'wav':
        wav_metadata = _validate_pcm_wav_frames(data, source)

    metadata = _probe_audio_with_pyav(data, source)
    if metadata is None:
        metadata = _probe_audio_with_ffprobe(data, source)
    if metadata is None:
        if wav_metadata is not None:
            return wav_metadata
        raise AudioNormalizationError(
            f'PyAV or ffprobe is required to detect sample rate, channels, and '
            f'channel layout for {source!r}.'
        )
    if wav_metadata is not None:
        if metadata.sample_rate != wav_metadata.sample_rate or metadata.channels != wav_metadata.channels:
            raise AudioNormalizationError(
                f'Conflicting WAV metadata detected for {source!r}: container reports '
                f'{wav_metadata.sample_rate} Hz/{wav_metadata.channels} channels but decoder '
                f'reports {metadata.sample_rate} Hz/{metadata.channels} channels.'
            )
    return metadata


def _validate_complete_audio_decode(path):
    """Decode every audio frame for publication without changing metadata probes."""
    try:
        import av
    except ImportError:
        av = None

    if av is not None:
        decoded_samples = 0
        try:
            with av.open(path, mode='r', options={'err_detect': 'explode'}) as container:
                streams = [stream for stream in container.streams if stream.type == 'audio']
                if not streams:
                    raise AudioNormalizationError(f'No audio stream found in {path!r}.')
                for frame in container.decode(streams[0]):
                    decoded_samples += max(0, int(getattr(frame, 'samples', 0)))
        except AudioNormalizationError:
            raise
        except Exception as err:
            raise AudioNormalizationError(
                f'Unable to completely decode audio from {path!r}: {err}'
            ) from err
        if decoded_samples <= 0:
            raise AudioNormalizationError(
                f'Audio stream in {path!r} contains no completely decodable frames.'
            )
        return

    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        return
    command = [
        ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-xerror',
        '-i', path, '-map', '0:a:0', '-f', 'null', '-',
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as err:
        raise AudioNormalizationError(
            f'Unable to completely decode audio from {path!r}: {err}'
        ) from err
    if result.returncode != 0:
        error = result.stderr.decode('utf-8', errors='replace').strip()
        raise AudioNormalizationError(
            f'Unable to completely decode audio from {path!r} with ffmpeg: {error}'
        )


def _validate_audio_file_for_publication(path):
    """Apply structural, metadata, and full-decode checks before publication."""
    if not is_reusable_audio_file(path):
        raise AudioNormalizationError(f'Audio file has an incomplete container: {path!r}.')
    validate_audio_file(path)
    _validate_complete_audio_decode(path)


def probe_audio_metadata(local_path, max_file_size=None):
    """Validate a local audio file and return metadata from its actual stream."""
    data, declared_format, source = read_audio_file(
        local_path, max_file_size=max_file_size
    )
    if not data:
        raise AudioNormalizationError('Audio source is empty.')
    detected_format = detect_audio_format(data)
    if detected_format is None:
        raise AudioNormalizationError(
            f'Unknown or unsupported audio container for source {source!r}.'
        )
    return _probe_audio_metadata(data, detected_format, source)


def validate_audio_file(path):
    """Return audio metadata for a valid local file, otherwise raise a specific error."""
    return probe_audio_metadata(path)


def is_valid_audio_file(path):
    try:
        validate_audio_file(path)
    except (AudioNormalizationError, OSError, ValueError):
        return False
    return True


def read_audio_file(local_path, max_file_size=None):
    """Read the local-path intermediate protocol used by audio normalization."""
    max_file_size = validate_audio_size_limit(max_file_size)
    if isinstance(local_path, os.PathLike):
        local_path = os.fspath(local_path)
    if not isinstance(local_path, str) or not local_path:
        raise AudioNormalizationError('Audio input must be a non-empty local path.')
    if local_path.lower().startswith(AUDIO_MEDIA_URL_PREFIXES):
        raise AudioNormalizationError(
            'Audio normalization accepts only local paths; call '
            'resolve_media_source() first.'
        )
    if not osp.isfile(local_path):
        raise AudioNormalizationError(f'Audio file does not exist: {local_path!r}.')
    suffix = suffix_from_media_value(local_path)
    declared_format = AUDIO_FORMAT_MAP.get(suffix)
    if declared_format is None:
        raise AudioNormalizationError(
            f'Unsupported or missing audio file extension: {suffix or "<none>"}.'
        )
    size = osp.getsize(local_path)
    check_audio_size(size, max_file_size, source=local_path)
    with open(local_path, 'rb') as f:
        data = f.read()
    return data, declared_format, local_path


def _transcode_audio_to_wav_pyav(
    data,
    source_format,
    source,
    sample_rate,
    channel_layout,
    max_output_file_size=None,
):
    try:
        import av
    except ImportError as err:
        raise AudioNormalizationError(
            f'ffmpeg or PyAV is required to normalize {source_format} audio to WAV, '
            'but neither is available.'
        ) from err

    try:
        input_buffer = io.BytesIO(data)
        output_buffer = _BoundedAudioBuffer(
            max_output_file_size,
            source=f'normalized audio payload from {source}',
        )
        with av.open(input_buffer, mode='r') as input_container:
            audio_streams = [stream for stream in input_container.streams if stream.type == 'audio']
            if not audio_streams:
                raise AudioNormalizationError(f'No audio stream found in {source!r}.')

            input_stream = audio_streams[0]
            with av.open(output_buffer, mode='w', format='wav') as output_container:
                output_stream = output_container.add_stream('pcm_s16le', rate=sample_rate)
                output_stream.layout = channel_layout
                resampler = av.audio.resampler.AudioResampler(
                    format='s16', layout=channel_layout, rate=sample_rate
                )
                for frame in input_container.decode(input_stream):
                    for normalized_frame in resampler.resample(frame):
                        for packet in output_stream.encode(normalized_frame):
                            output_container.mux(packet)
                for normalized_frame in resampler.resample(None):
                    for packet in output_stream.encode(normalized_frame):
                        output_container.mux(packet)
                for packet in output_stream.encode(None):
                    output_container.mux(packet)
        return output_buffer.getvalue()
    except AudioNormalizationError:
        raise
    except Exception as err:
        raise AudioNormalizationError(
            f'PyAV failed to normalize {source_format} audio from {source!r}: {err}'
        ) from err


@lru_cache(maxsize=8)
def _ffmpeg_backend_identity(ffmpeg):
    try:
        stat = os.stat(ffmpeg)
        return f'ffmpeg:{osp.realpath(ffmpeg)}:{stat.st_size}:{stat.st_mtime_ns}'
    except OSError:
        return f'ffmpeg:{ffmpeg}'


def _pyav_backend_identity():
    try:
        import av
    except ImportError as err:
        raise AudioNormalizationError(
            'ffmpeg or PyAV is required to normalize audio to WAV, but neither is available.'
        ) from err
    versions = getattr(av, 'library_versions', {})
    libraries = ','.join(f'{name}={version}' for name, version in sorted(versions.items()))
    return f'pyav:{av.__version__}:{libraries}'


def _transcode_backend():
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is not None:
        return 'ffmpeg', ffmpeg, _ffmpeg_backend_identity(ffmpeg)
    return 'pyav', None, _pyav_backend_identity()


def _valid_cached_wav(path, sample_rate, channels):
    try:
        with open(path, 'rb') as f:
            data = f.read()
        if detect_audio_format(data) != 'wav':
            return None
        metadata = _probe_audio_metadata(data, 'wav', path)
        if metadata.sample_rate != sample_rate or metadata.channels != channels:
            return None
        return data, metadata
    except AudioNormalizationError:
        # A cache entry is an optimization, not the source of truth. Treat
        # malformed or truncated entries as misses so the original source can
        # be transcoded again. The caller checks the active size policy first.
        return None
    except OSError:
        return None


def _transcode_audio_to_wav(
    data,
    source_format,
    source,
    cache_dir,
    sample_rate,
    channels,
    channel_layout,
    max_output_file_size=None,
):
    max_output_file_size = validate_audio_size_limit(max_output_file_size)
    backend_kind, backend_path, backend_identity = _transcode_backend()
    policy = (
        f'pcm_s16le:sample_rate={sample_rate}:channels={channels}:'
        f'channel_layout={channel_layout}:backend={backend_identity}'
    )
    cache_key = hashlib.sha256(policy.encode() + b'\0' + data).hexdigest()
    target = osp.join(cache_dir, AUDIO_CACHE_PREFIX + cache_key + '.wav')
    if osp.isfile(target):
        try:
            cached_size = osp.getsize(target)
        except OSError:
            cached = None
        else:
            check_audio_size(
                cached_size,
                max_output_file_size,
                source=f'cached normalized audio payload {target}',
            )
            cached = _valid_cached_wav(target, sample_rate, channels)
        if cached is not None:
            os.utime(target, None)
            return cached[0], cached[1], True
        logger.warning(f'Removing invalid audio cache entry: {target}')
        try:
            os.unlink(target)
        except OSError:
            pass

    os.makedirs(cache_dir, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        prefix=f'.{cache_key}.', suffix='.wav', dir=cache_dir, delete=False
    )
    tmp_path = tmp.name
    tmp.close()
    try:
        if backend_kind == 'pyav':
            normalized = _transcode_audio_to_wav_pyav(
                data=data,
                source_format=source_format,
                source=source,
                sample_rate=sample_rate,
                channel_layout=channel_layout,
                max_output_file_size=max_output_file_size,
            )
            check_audio_size(
                len(normalized),
                max_output_file_size,
                source=f'normalized audio payload from {source}',
            )
            with open(tmp_path, 'wb') as f:
                f.write(normalized)
        else:
            command = [
                backend_path, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                '-i', 'pipe:0', '-map_metadata', '-1', '-vn', '-c:a', 'pcm_s16le',
                '-ar', str(sample_rate), '-ac', str(channels),
            ]
            if max_output_file_size is not None:
                command.extend(['-fs', str(max_output_file_size + 1)])
            command.append(tmp_path)
            result = subprocess.run(
                command,
                input=data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=300,
            )
            if result.returncode != 0:
                error = result.stderr.decode('utf-8', errors='replace').strip()
                raise AudioNormalizationError(
                    f'ffmpeg failed to normalize {source_format} audio from {source!r}: {error}'
                )
            check_audio_size(
                osp.getsize(tmp_path),
                max_output_file_size,
                source=f'normalized audio payload from {source}',
            )
            with open(tmp_path, 'rb') as f:
                normalized = f.read()
        metadata = _probe_audio_metadata(normalized, 'wav', target)
        if metadata.sample_rate != sample_rate or metadata.channels != channels:
            raise AudioNormalizationError(
                f'Audio normalization produced {metadata.sample_rate} Hz/{metadata.channels} '
                f'channels instead of {sample_rate} Hz/{channels} channels for {source!r}.'
            )
        check_audio_size(
            len(normalized),
            max_output_file_size,
            source=f'normalized audio payload from {source}',
        )
        os.replace(tmp_path, target)
    except subprocess.TimeoutExpired as err:
        raise AudioNormalizationError(
            f'ffmpeg timed out while normalizing audio from {source!r}.'
        ) from err
    except OSError as err:
        raise AudioNormalizationError(
            f'Failed to execute ffmpeg while normalizing audio from {source!r}: {err}'
        ) from err
    finally:
        if osp.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return normalized, metadata, False


def _audio_policy(value, name):
    if value is None or value == 'auto':
        return 'auto'
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AudioNormalizationError(f'{name} must be "auto" or a positive integer.')
    if name == 'channels' and value not in (1, 2):
        raise AudioNormalizationError('channels must be "auto", 1, or 2.')
    return value


def default_audio_cache_dir():
    configured = os.environ.get('VLMEVAL_AUDIO_CACHE')
    if configured:
        return configured
    return osp.join(LMUDataRoot(), 'audio_cache')


def normalize_audio(
    local_path,
    target_format='wav',
    cache_dir=None,
    sample_rate='auto',
    channels='auto',
    max_file_size=None,
    max_output_file_size=None,
):
    """Validate a local audio file and optionally normalize it to cached PCM WAV.

    ``local_path`` is the intermediate protocol produced by
    :func:`resolve_media_source`. The file extension and byte signature are
    compared; the byte signature is authoritative and a mismatched extension
    produces a warning.

    ``sample_rate='auto'`` and ``channels='auto'`` preserve detected source values;
    ``None`` remains a backwards-compatible alias for ``'auto'``. Explicit values
    force WAV-to-WAV conversion when the detected input does not match. They
    are part of the versioned cache key together with the active backend version.
    Set ``target_format=None`` to validate without transcoding.
    ``max_file_size`` limits the input while ``max_output_file_size`` separately
    limits the final provider payload after any WAV transcoding.
    """
    if target_format not in (None, 'wav'):
        raise AudioNormalizationError(f'Unsupported target audio format: {target_format!r}.')
    sample_rate_policy = _audio_policy(sample_rate, 'sample_rate')
    channels_policy = _audio_policy(channels, 'channels')
    max_output_file_size = validate_audio_size_limit(max_output_file_size)

    data, declared_format, source = read_audio_file(
        local_path, max_file_size=max_file_size
    )
    if not data:
        raise AudioNormalizationError('Audio source is empty.')
    detected_format = detect_audio_format(data)
    if detected_format is None:
        raise AudioNormalizationError(
            f'Unknown or unsupported audio container for source {source!r}.'
        )
    if not _compatible_audio_formats(declared_format, detected_format):
        mismatch = (
            f'Audio format mismatch for {source!r}: declared {declared_format}, '
            f'content is {detected_format}.'
        )
        logger.warning(f'{mismatch} Trusting the detected local container.')

    source_format = detected_format
    source_metadata = _probe_audio_metadata(data, source_format, source)
    sent_sample_rate = (
        source_metadata.sample_rate if sample_rate_policy == 'auto' else sample_rate_policy
    )
    sent_channels = source_metadata.channels if channels_policy == 'auto' else channels_policy
    sent_layout = source_metadata.channel_layout
    if channels_policy == 1:
        sent_layout = 'mono'
    elif channels_policy == 2:
        sent_layout = 'stereo'

    needs_transcode = target_format == 'wav' and (
        source_format != 'wav'
        or sent_sample_rate != source_metadata.sample_rate
        or sent_channels != source_metadata.channels
        or (
            channels_policy in (1, 2)
            and source_metadata.channel_layout != sent_layout
        )
    )
    if target_format is None or not needs_transcode:
        mime_type = AUDIO_FORMAT_MIME_MAP[source_format]
        check_audio_size(
            len(data), max_output_file_size, source=f'final audio payload from {source}'
        )
        logger.info(
            f'Validated audio: original_mime={mime_type}, sent_mime={mime_type}, '
            f'bytes={len(data)}, duration_seconds={source_metadata.duration}, transcoded=False, '
            f'source_sample_rate={source_metadata.sample_rate}, '
            f'sent_sample_rate={source_metadata.sample_rate}, '
            f'source_channels={source_metadata.channels}, sent_channels={source_metadata.channels}, '
            f'source_channel_layout={source_metadata.channel_layout}, '
            f'sent_channel_layout={source_metadata.channel_layout}'
        )
        return AudioPayload(
            data=data,
            mime_type=mime_type,
            format=source_format,
            source=source,
            source_metadata=source_metadata,
            sent_metadata=source_metadata,
        )

    if cache_dir is None:
        cache_dir = default_audio_cache_dir()
    normalized, sent_metadata, cache_hit = _transcode_audio_to_wav(
        data=data,
        source_format=source_format,
        source=source,
        cache_dir=os.fspath(cache_dir),
        sample_rate=sent_sample_rate,
        channels=sent_channels,
        channel_layout=sent_layout,
        max_output_file_size=max_output_file_size,
    )
    check_audio_size(
        len(normalized), max_output_file_size, source=f'normalized audio payload from {source}'
    )
    logger.info(
        f'Normalized audio: original_mime={AUDIO_FORMAT_MIME_MAP[source_format]}, '
        f'sent_mime=audio/wav, bytes={len(data)}, duration_seconds={sent_metadata.duration}, '
        f'transcoded=True, cache_hit={cache_hit}, '
        f'source_sample_rate={source_metadata.sample_rate}, '
        f'sent_sample_rate={sent_metadata.sample_rate}, '
        f'source_channels={source_metadata.channels}, sent_channels={sent_metadata.channels}, '
        f'source_channel_layout={source_metadata.channel_layout}, '
        f'sent_channel_layout={sent_metadata.channel_layout}'
    )
    return AudioPayload(
        data=normalized,
        mime_type='audio/wav',
        format='wav',
        source=source,
        source_metadata=source_metadata,
        sent_metadata=sent_metadata,
    )


def prune_audio_cache(cache_dir, max_size_bytes=None, max_age_seconds=None, now=None):
    """Remove expired/least-recently-used v2 transcode entries from a cache.

    Cache hits refresh ``mtime``. Supplying ``max_age_seconds`` removes older
    entries first; supplying ``max_size_bytes`` then removes the least recently
    used entries until the requested capacity is met. Other files are untouched.
    """
    cache_dir = os.fspath(cache_dir)
    if max_size_bytes is None and max_age_seconds is None:
        raise ValueError('At least one cache cleanup boundary must be provided.')
    for value, name in ((max_size_bytes, 'max_size_bytes'), (max_age_seconds, 'max_age_seconds')):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
        ):
            raise ValueError(f'{name} must be a non-negative number or None.')
    if not osp.isdir(cache_dir):
        return {'removed_files': 0, 'removed_bytes': 0, 'remaining_bytes': 0}

    entries = []
    for name in os.listdir(cache_dir):
        if not name.startswith(AUDIO_CACHE_PREFIX) or not name.endswith('.wav'):
            continue
        path = osp.join(cache_dir, name)
        try:
            stat = os.stat(path)
        except OSError:
            continue
        if osp.isfile(path):
            entries.append([path, stat.st_mtime, stat.st_size])

    now = time.time() if now is None else now
    removed_files = 0
    removed_bytes = 0

    def remove(entry):
        nonlocal removed_files, removed_bytes
        try:
            os.unlink(entry[0])
        except OSError:
            return False
        removed_files += 1
        removed_bytes += entry[2]
        return True

    remaining = []
    for entry in entries:
        expired = max_age_seconds is not None and now - entry[1] > max_age_seconds
        if not expired or not remove(entry):
            remaining.append(entry)

    total = sum(entry[2] for entry in remaining)
    if max_size_bytes is not None and total > max_size_bytes:
        for entry in sorted(remaining, key=lambda item: item[1]):
            if total <= max_size_bytes:
                break
            if remove(entry):
                total -= entry[2]

    return {
        'removed_files': removed_files,
        'removed_bytes': removed_bytes,
        'remaining_bytes': total,
    }


__all__ = [
    'AUDIO_CACHE_PREFIX',
    'AUDIO_FORMAT_EXTENSION_MAP',
    'AUDIO_FORMAT_MAP',
    'AUDIO_FORMAT_MIME_MAP',
    'AUDIO_FORMAT_REGISTRY',
    'AUDIO_IO_CHUNK_SIZE',
    'AUDIO_MEDIA_URL_PREFIXES',
    'AUDIO_MIME_ALIASES',
    'AUDIO_MIME_FORMAT_MAP',
    'AUDIO_MIME_MAP',
    'AudioFormatSpec',
    'AudioDownloadError',
    'AudioMetadata',
    'AudioNormalizationError',
    'AudioPayload',
    'AudioSizeLimitError',
    'UnsupportedAudioSourceError',
    'atomic_write_audio_file',
    'audio_mime_type',
    'base64_decoded_size',
    'check_audio_size',
    'default_audio_cache_dir',
    'detect_audio_format',
    'infer_audio_mime_type',
    'is_audio_media_url',
    'is_reusable_audio_file',
    'is_valid_audio_file',
    'normalize_audio',
    'probe_audio_metadata',
    'prune_audio_cache',
    'read_audio_file',
    'resolve_media_source',
    'suffix_from_media_value',
    'validate_audio_file',
    'validate_audio_size_limit',
]
