#!/usr/bin/env python

import getpass
import subprocess
from pathlib import Path

USERNAME = getpass.getuser()
SSHDIR = Path.home() / ".ssh"
KEY_PATH = SSHDIR / "id_rsa"


def generate_key():
    subprocess.run(
        ["ssh-keygen", "-t", "rsa", "-f", str(KEY_PATH), "-N", ""],
    )


def authorize_key():
    public_key_path = KEY_PATH.with_suffix(".pub")
    authorized_keys = SSHDIR / "authorized_keys"

    print(f"Copying contents in {public_key_path} to {authorized_keys}...")

    with public_key_path.open() as keyfile, authorized_keys.open("a+") as authfile:
        authfile.write(keyfile.read())


def add_hosts():
    sshcfg = SSHDIR / "config"

    print("Writing host configurations...")

    with sshcfg.open("a+") as file:
        for i, last in zip(range(1, 10), range(72, 81)):
            print(f"Host pp{i}", file=file)
            print(f"\tHostName 192.168.108.{last}", file=file)
            print(f"\tUser {USERNAME}", file=file)
            print(file=file)


def main() -> None:
    SSHDIR.mkdir(parents=True, exist_ok=True)
    generate_key()
    authorize_key()
    add_hosts()


if __name__ == "__main__":
    main()
