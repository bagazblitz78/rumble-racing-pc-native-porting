"""Test fixture: builds a tiny PS2-style disc image (raw MODE2/2352 + ISO9660).

Layout (LBAs):
  0-15   system area (zero-filled)
  16     Primary Volume Descriptor
  17     Supplementary Volume Descriptor (PLAYSTATION, like real PS2 discs)
  18     Volume Descriptor Set Terminator
  19     unused (zero)
  20     root directory extent
  21     SUB/ directory extent
  22     HELLO.TXT data
  23     SUB/INNER.TXT data
"""
import struct

from ps2toolkit.disc.sector import RAW_SECTOR_SIZE, SYNC_PATTERN, encode_bcd

TOTAL_LBAS = 24
ROOT_LBA = 20
SUB_LBA = 21
HELLO_LBA = 22
INNER_LBA = 23

HELLO_CONTENT = b"Hello, PS2!\n"
INNER_CONTENT = b"inner file content\n"

ISO_BLOCK = 2048


def both32(value: int) -> bytes:
    return struct.pack("<I", value) + struct.pack(">I", value)


def both16(value: int) -> bytes:
    return struct.pack("<H", value) + struct.pack(">H", value)


def dir_record(name: bytes, lba: int, size: int, is_dir: bool) -> bytes:
    name_len = len(name)
    length = 33 + name_len + (1 if name_len % 2 == 0 else 0)
    rec = bytearray(length)
    rec[0] = length
    rec[1] = 0  # extended attributes: none
    rec[2:10] = both32(lba)
    rec[10:18] = both32(size)
    # recording date (7 bytes) left zero
    rec[25] = 0x02 if is_dir else 0x00
    rec[28:32] = both16(1)  # volume sequence number
    rec[32] = name_len
    rec[33 : 33 + name_len] = name
    return bytes(rec)


def directory_extent(self_lba: int, parent_lba: int, entries: list[bytes]) -> bytes:
    data = bytearray()
    data += dir_record(b"\x00", self_lba, ISO_BLOCK, is_dir=True)
    data += dir_record(b"\x01", parent_lba, ISO_BLOCK, is_dir=True)
    for entry in entries:
        data += entry
    assert len(data) <= ISO_BLOCK
    return bytes(data).ljust(ISO_BLOCK, b"\x00")


def volume_descriptor(
    vd_type: int, system_id: bytes, volume_id: bytes, root_lba: int = ROOT_LBA, total_lbas: int = TOTAL_LBAS
) -> bytes:
    sector = bytearray(ISO_BLOCK)
    sector[0] = vd_type
    sector[1:6] = b"CD001"
    sector[6] = 1  # version
    sector[8 : 8 + len(system_id)] = system_id
    sector[8 + len(system_id) : 40] = b" " * (40 - 8 - len(system_id))
    sector[40 : 40 + len(volume_id)] = volume_id
    sector[40 + len(volume_id) : 72] = b" " * (72 - 40 - len(volume_id))
    sector[80:88] = both32(total_lbas)  # volume space size (in blocks)
    sector[120:124] = both16(1)  # volume set size
    sector[124:128] = both16(1)  # volume sequence number
    sector[128:132] = both16(ISO_BLOCK)  # logical block size
    sector[156:190] = dir_record(b"\x00", root_lba, ISO_BLOCK, is_dir=True)
    return bytes(sector)


def make_raw_sector(lba: int, user_data: bytes) -> bytes:
    mm, ss, ff = 0, 2 + lba // 75, lba % 75
    header = bytes([encode_bcd(mm), encode_bcd(ss), encode_bcd(ff), 2])
    submode = 0x08  # data, Form1
    subheader = bytes([0, 0, submode, 0, 0, 0, submode, 0])
    payload = subheader + user_data[:2048].ljust(ISO_BLOCK, b"\x00") + b"\x00" * (4 + 276)
    sector = SYNC_PATTERN + header + payload
    assert len(sector) == RAW_SECTOR_SIZE
    return sector


def build_test_disc(pvd_system: bytes = b"LINUX", svd_system: bytes = b"PLAYSTATION") -> bytes:
    blocks: dict[int, bytes] = {
        16: volume_descriptor(1, pvd_system, b"TESTVOL"),
        17: volume_descriptor(2, svd_system, b"TESTVOL"),
        18: volume_descriptor(0, b"", b""),  # terminator (type 0, with CD001)
        ROOT_LBA: directory_extent(
            ROOT_LBA, ROOT_LBA, [dir_record(b"HELLO.TXT;1", HELLO_LBA, len(HELLO_CONTENT), False),
                                 dir_record(b"SUB", SUB_LBA, ISO_BLOCK, True)]
        ),
        SUB_LBA: directory_extent(
            SUB_LBA, ROOT_LBA, [dir_record(b"INNER.TXT;1", INNER_LBA, len(INNER_CONTENT), False)]
        ),
        HELLO_LBA: HELLO_CONTENT,
        INNER_LBA: INNER_CONTENT,
    }
    out = bytearray()
    for lba in range(TOTAL_LBAS):
        out += make_raw_sector(lba, blocks.get(lba, b""))
    return bytes(out)


# --- boot-disc fixture (SYSTEM.CNF + boot ELF) ---
BOOT_ROOT_LBA = 20
BOOT_CNF_LBA = 21
BOOT_LBA = 22
BOOT_TOTAL_LBAS = 26

FAKE_ELF = b"\x7fELF" + bytes([0x01, 0x02, 0x01, 0x00]) + b"\x98\x83\x00\x00" + b"R" * 120

BOOT_CNF_CONTENT = b"BOOT2 = cdrom0:\\SLUS_201.74;1\nVER = 1.00\n"


def build_disc_with_boot(elf_offset: int = 0) -> bytes:
    """Disc with SYSTEM.CNF;1 and SLUS_201.74;1 (fake ELF at elf_offset)."""
    boot_size = elf_offset + len(FAKE_ELF)
    boot_data = bytearray(boot_size)
    boot_data[elf_offset : elf_offset + len(FAKE_ELF)] = FAKE_ELF
    blocks: dict[int, bytes] = {
        16: volume_descriptor(1, b"LINUX", b"BOOTVOL", root_lba=BOOT_ROOT_LBA, total_lbas=BOOT_TOTAL_LBAS),
        17: bytes([0]) + b"CD001" + bytes(ISO_BLOCK - 6),  # terminator
        BOOT_ROOT_LBA: directory_extent(
            BOOT_ROOT_LBA,
            BOOT_ROOT_LBA,
            [
                dir_record(b"SYSTEM.CNF;1", BOOT_CNF_LBA, len(BOOT_CNF_CONTENT), False),
                dir_record(b"SLUS_201.74;1", BOOT_LBA, boot_size, False),
            ],
        ),
        BOOT_CNF_LBA: BOOT_CNF_CONTENT,
    }
    for i in range(0, len(boot_data), ISO_BLOCK):
        blocks[BOOT_LBA + i // ISO_BLOCK] = boot_data[i : i + ISO_BLOCK]
    out = bytearray()
    for lba in range(BOOT_TOTAL_LBAS):
        out += make_raw_sector(lba, blocks.get(lba, b""))
    return bytes(out)
