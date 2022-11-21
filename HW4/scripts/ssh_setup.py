#!/usr/bin/env python

import getpass
import subprocess
from pathlib import Path

import pexpect

SSHDIR = Path.home() / ".ssh"
KEY_PATH = SSHDIR / "id_rsa"


def available_hosts():
    for i in range(2, 11):
        if i == 9:
            continue
        yield i


def generate_key():
    subprocess.run(
        ["ssh-keygen", "-t", "rsa", "-f", str(KEY_PATH), "-N", ""],
    )


def add_hosts():
    username = getpass.getuser()
    sshcfg = SSHDIR / "config"

    print("Writing host configurations...")

    with sshcfg.open("a+") as file:
        for i in available_hosts():
            print(f"Host pp{i}", file=file)
            print(f"\tHostName 192.168.202.{i}", file=file)
            print(f"\tUser {username}", file=file)
            print(f"\tIdentitiesOnly yes", file=file)
            print(f"\tIdentityFile {KEY_PATH}", file=file)
            print(file=file)


def copy_keys(password: str):
    public_key_path = KEY_PATH.with_suffix(".pub")
    authorized_keys = SSHDIR / "authorized_keys"

    for i in available_hosts():
        print(f"Copying public key into pp{i}")

        cmd = f"cat {public_key_path} | ssh -o StrictHostKeyChecking=accept-new pp{i} 'mkdir -p {SSHDIR} && cat >> {authorized_keys}'"

        connection = pexpect.spawn(
            "/bin/bash",
            ["-c", cmd],
            encoding="utf-8",
        )

        connection.expect("[pP]assword: ")
        connection.sendline(password)

        connection.expect(pexpect.EOF)

        connection.close()


def main() -> None:
    password = getpass.getpass("SSH connection password: ")
    SSHDIR.mkdir(parents=True, exist_ok=True)
    generate_key()
    add_hosts()
    copy_keys(password)


if __name__ == "__main__":
    main()
