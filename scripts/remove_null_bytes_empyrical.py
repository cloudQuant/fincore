from pathlib import Path


if __name__ == "__main__":
    path = Path("fincore/empyrical.py")
    data = path.read_bytes()
    null_count_before = data.count(b"\x00")
    print("null_count_before", null_count_before)

    if null_count_before == 0:
        print("No null bytes found. Nothing to do.")
    else:
        cleaned = data.replace(b"\x00", b"")
        path.write_bytes(cleaned)
        print("null_count_after", cleaned.count(b"\x00"))
        print("Cleaned null bytes from", path)
