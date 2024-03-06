

def voice_vits_streaming(text, id=0, format="wav", lang="auto", length=1, noise=0.667, noisew=0.8, segment_size=50,
                         save_audio=True, save_path=None):
    fields = {
        "text": text,
        "id": str(id),
        "format": format,
        "lang": lang,
        "length": str(length),
        "noise": str(noise),
        "noisew": str(noisew),
        "segment_size": str(segment_size),
        "streaming": 'True'
    }
    boundary = '----VoiceConversionFormBoundary' + ''.join(random.sample(string.ascii_letters + string.digits, 16))

    m = MultipartEncoder(fields=fields, boundary=boundary)
    headers = {"Content-Type": m.content_type}
    url = f"{base_url}/voice"

    res = requests.post(url=url, data=m, headers=headers)
    fname = re.findall("filename=(.+)", res.headers["Content-Disposition"])[0]
    if save_path is not None:
        path = os.path.join(save_path, fname)
    else:
        path = os.path.join(absolute_path, fname)
    if save_audio:
        with open(path, "wb") as f:
            f.write(res.content)
        print(path)
        return path
    return None