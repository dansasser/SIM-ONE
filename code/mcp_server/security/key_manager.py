import os
import json
import hashlib
from typing import List, Dict, Optional
from datetime import datetime

API_KEYS_FILE = "api_keys.json"
SALT = os.urandom(16).hex()  # Generate a random salt

def hash_api_key(api_key: str, salt: str) -> str:
    """Hashes an API key with a salt using SHA-256."""
    hasher = hashlib.sha256()
    hasher.update(salt.encode('utf-8'))
    hasher.update(api_key.encode('utf-8'))
    return hasher.hexdigest()

def load_api_keys() -> List[Dict[str, str]]:
    """Loads API keys from the JSON file."""
    if not os.path.exists(API_KEYS_FILE):
        return []
    try:
        with open(API_KEYS_FILE, "r") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return []

def save_api_keys(keys: List[Dict[str, str]]) -> None:
    """Saves API keys to the JSON file."""
    with open(API_KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=2)

def add_api_key(api_key: str, role: str, user_id: str) -> None:
    """Adds a new API key to the store."""
    keys = load_api_keys()
    hashed_key = hash_api_key(api_key, SALT)
    keys.append({
        "hash": hashed_key,
        "role": role,
        "user_id": user_id,
        "salt": SALT,
        "created_at": datetime.utcnow().isoformat() + "Z"
    })
    save_api_keys(keys)

def remove_api_key_by_user_id(user_id: str) -> bool:
    """Removes API key(s) for a given user_id. Returns True if any removed."""
    keys = load_api_keys()
    before = len(keys)
    keys = [k for k in keys if k.get("user_id") != user_id]
    if len(keys) < before:
        save_api_keys(keys)
        return True
    return False

def validate_api_key(api_key: str) -> Optional[Dict[str, str]]:
    """
    Validates an API key.
    Returns the key's role and user_id if valid, otherwise None.
    """
    if not api_key:
        return None

    stored_keys = load_api_keys()
    for key_data in stored_keys:
        # Each key could have a different salt
        salt = key_data.get("salt", SALT)
        hashed_key = hash_api_key(api_key, salt)
        if hashed_key == key_data["hash"]:
            return {"role": key_data["role"], "user_id": key_data["user_id"]}

    return None

def initialize_api_keys():
    """
    Initializes the API key store from environment variables.
    This should be run once on application startup.
    """
    if os.path.exists(API_KEYS_FILE):
        return

    api_keys_str = os.getenv("VALID_API_KEYS", "")
    if not api_keys_str:
        print("WARNING: No VALID_API_KEYS found in environment. No API keys will be active.")
        return

    keys_to_store = []
    # For simplicity in this example, we'll assign a default 'user' role.
    # In a real system, you might have more complex logic to assign roles.
    for key in api_keys_str.split(','):
        key = key.strip()
        if not key:
            continue

        # This is a sample role assignment.
        # For a real production system, you would have a more secure way to assign roles.
        if "admin" in key:
            role = "admin"
        elif "readonly" in key:
            role = "read-only"
        else:
            role = "user"

        user_id = f"user_{hashlib.sha1(key.encode()).hexdigest()[:8]}"

        salt = os.urandom(16).hex()
        hashed_key = hash_api_key(key, salt)
        keys_to_store.append({
            "hash": hashed_key,
            "role": role,
            "user_id": user_id,
            "salt": salt
        })

    save_api_keys(keys_to_store)
    print(f"Initialized {len(keys_to_store)} API keys from environment variables.")

if __name__ == '__main__':
    # Example usage:
    # This part is for testing and won't be executed when imported.
    print("Running key manager setup...")

    # In a real app, you would get this from a secure source, not hardcode it.
    os.environ['VALID_API_KEYS'] = 'test-key-admin,test-key-user'

    initialize_api_keys()

    print("\nValidating keys...")
    print("test-key-admin:", validate_api_key('test-key-admin'))
    print("test-key-user:", validate_api_key('test-key-user'))
    print("invalid-key:", validate_api_key('invalid-key'))

    # Clean up the created file
    if os.path.exists(API_KEYS_FILE):
        os.remove(API_KEYS_FILE)
    print(f"\nCleaned up {API_KEYS_FILE}.")
