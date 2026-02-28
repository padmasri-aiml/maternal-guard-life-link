# ======================================================
# SECURITY UTILITIES
# Maternal-Guard & Life-Link
# ======================================================

from cryptography.fernet import Fernet
import os

# ------------------------------------------------------
# Encryption key file
# ------------------------------------------------------
KEY_PATH = "secret.key"


def load_key():
    """Load existing key or create a new one."""
    if not os.path.exists(KEY_PATH):
        key = Fernet.generate_key()
        with open(KEY_PATH, "wb") as f:
            f.write(key)
    else:
        with open(KEY_PATH, "rb") as f:
            key = f.read()
    return key


cipher = Fernet(load_key())


# ------------------------------------------------------
# Encrypt medical data (stored in DB)
# ------------------------------------------------------
def encrypt_data(text: str) -> str:
    if text is None or text == "":
        return ""
    return cipher.encrypt(text.encode()).decode()


# ------------------------------------------------------
# Decrypt (ADMIN ONLY — not used in UI)
# ------------------------------------------------------
def decrypt_data(token: str) -> str:
    try:
        return cipher.decrypt(token.encode()).decode()
    except Exception:
        return "Protected Medical Record"


# ------------------------------------------------------
# Mask data for normal users
# ------------------------------------------------------
def mask_medical_data(_):
    """Never expose medical history in UI."""
    return "Protected Medical Record"