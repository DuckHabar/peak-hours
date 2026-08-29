import requests, time, sys, os
PK = "https://disk.yandex.ru/d/DkzPt3tNEokDvg"
API = "https://cloud-api.yandex.net/v1/disk/public/resources"

def req(url, params, tries=6):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=90)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(5*(i+1)); continue
            r.raise_for_status()
        except Exception as e:
            if i == tries-1: raise
            time.sleep(3*(i+1))
    return None

def ls(path, limit=1000):
    items = []
    offset = 0
    while True:
        d = req(API, {"public_key": PK, "path": path, "limit": limit, "offset": offset})
        emb = d.get('_embedded', {})
        batch = emb.get('items', [])
        items.extend(batch)
        total = emb.get('total', 0)
        offset += len(batch)
        if offset >= total or not batch: break
    return items

def download(path, dest):
    d = req(API + "/download", {"public_key": PK, "path": path})
    href = d['href']
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for i in range(5):
        try:
            with requests.get(href, stream=True, timeout=180) as r:
                r.raise_for_status()
                with open(dest, 'wb') as f:
                    for chunk in r.iter_content(1<<20):
                        f.write(chunk)
            return dest
        except Exception:
            time.sleep(3*(i+1))
    raise RuntimeError(f"failed {path}")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "ls":
        for it in ls(sys.argv[2]):
            print(f"[{it['type']}] {it['name']}\t{it.get('size','')}")
    elif cmd == "get":
        print(download(sys.argv[2], sys.argv[3]))
