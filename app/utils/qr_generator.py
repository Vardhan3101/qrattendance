from __future__ import annotations

from pathlib import Path

import qrcode


def generate_user_qr(user_id: int, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = str(user_id)
    file_path = output_dir / f"user_{user_id}.png"
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(payload)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(file_path)
    return str(file_path)
