import argparse
import requests
import json

URL = 'http://127.0.0.1:8000/transcribe'


def parse_args():
    parser = argparse.ArgumentParser(description='Audio translator')
    parser.add_argument('-f', '--filenames', nargs='+', default=[], help='List of audio files to be transcribed')
    return parser.parse_args()


def upload_files(files):
    files_to_upload = [('files', open(f, 'rb')) for f in files]
    try:
        response = requests.post(URL, files=files_to_upload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return {}


def main():
    args = parse_args()
    if not args.filenames:
        print("No files provided for transcription.")
        return
    
    response_json = upload_files(args.filenames)
    if response_json:
        print(json.dumps(response_json, indent=4, ensure_ascii=False))
    else:
        print("Transcription failed.")


if __name__ == '__main__':
    main()
