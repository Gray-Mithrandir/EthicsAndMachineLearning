"""Download archives"""
import hashlib
import shutil
import sys

import requests
from tqdm import tqdm

from config import settings

links = [
    "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
    "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
    "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
    "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
    "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
    "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
    "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",
    "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
    "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
    "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
    "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",
    "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",
]

hashes = [
    "fe8ed0a6961412fddcbb3603c11b3698",
    "ab07a2d7cbe6f65ddd97b4ed7bde10bf",
    "2301d03bde4c246388bad3876965d574",
    "9f1b7f5aae01b13f4bc8e2c44a4b8ef6",
    "1861f3cd0ef7734df8104f2b0309023b",
    "456b53a8b351afd92a35bc41444c58c8",
    "1075121ea20a137b87f290d6a4a5965e",
    "b61f34cec3aa69f295fbb593cbd9d443",
    "442a3caa61ae9b64e61c561294d1e183",
    "09ec81c4c31e32858ad8cf965c494b74",
    "499aefc67207a5a97692424cf5dbeed5",
    "dc9fda1757c2de0032b63347a7d2895c",
]


def download(file_url, temp_file, file_num):
    """Download file"""
    r = requests.get(file_url, stream=True, timeout=60)
    with open(temp_file, "wb") as dl_fh:
        pbar = tqdm(
            unit="B", unit_scale=True, total=int(r.headers["Content-Length"]), desc=f"Archive {file_num+1}/{len(links)}"
        )
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                dl_fh.write(chunk)


def is_hashsum_valid(stored_hash, temp_file):
    """Return if file hash is valid"""
    with open(temp_file, "rb") as dl_fh:
        current_hash = hashlib.md5(dl_fh.read()).hexdigest()
        return stored_hash == current_hash


def main():
    """Main"""
    for archive_id, (file_url, file_hash) in enumerate(zip(links, hashes)):
        temp_file = settings.folders.root_folder / f"archive_{archive_id}.tar.gz"
        if temp_file.exists():
            print("File already exist. Checking if valid")
            if is_hashsum_valid(file_hash, temp_file):
                print("File is good. Extracting")
                shutil.unpack_archive(temp_file, settings.folders.root_folder)
                continue
            print("File hashsum is bad. Restart downloading")
            temp_file.unlink()
        print("Downloading")
        download(file_url, temp_file, archive_id)
        print("Validating")
        if not is_hashsum_valid(file_hash, temp_file):
            print("ERROR! File hash not match")
            sys.exit(-1)
        print("Extracting")
        shutil.unpack_archive(temp_file, settings.folders.root_folder)
    print("Removing temp files")
    for archive_id, _ in enumerate(links):
        temp_file = settings.folders.root_folder / f"archive_{archive_id}.tar.gz"
        temp_file.unlink()


if __name__ == "__main__":
    main()
