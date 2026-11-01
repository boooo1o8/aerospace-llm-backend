"""Black-box checks against the currently deployed inference service."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--key-file', type=Path, help='Private inference API key file')
    args = parser.parse_args()
    base = args.base_url.rstrip('/')
    key = os.environ.get('LLAMAFACTORY_API_KEY')
    if not key and args.key_file:
        key = args.key_file.read_text().strip()
    if not key:
        parser.error('Set LLAMAFACTORY_API_KEY or pass --key-file')
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'base_url': base}
    headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}

    def request(path, body=None, authenticated=True):
        return urllib.request.urlopen(urllib.request.Request(base + path,
            data=None if body is None else json.dumps(body).encode('utf-8'),
            headers=headers if authenticated else {}), timeout=90)

    def status_error(path, expected, body=None, authenticated=True):
        try:
            with request(path, body, authenticated):
                raise AssertionError(f'{path}: expected HTTP {expected}')
        except urllib.error.HTTPError as error:
            assert error.code == expected, (path, error.code, expected)

    status_error('/v1/models', 401, authenticated=False)
    report['unauthenticated_request'] = 'passed: 401'
    with request('/v1/models') as response:
        models = json.load(response)
    assert models['object'] == 'list' and models['data']
    model = models['data'][0]['id']
    report['model'] = model
    status_error('/v1/chat/completions', 422, body={'model': model})
    report['invalid_request'] = 'passed: 422'
    status_error('/v1/score/evaluation', 405, body={'model': model, 'messages': ['test']})
    report['reward_score_not_supported'] = 'passed: 405'
    body = {'model': model, 'messages': [{'role': 'user', 'content': '计算 17 加 25，简短回答。'}],
            'max_tokens': 32, 'temperature': 0.1, 'stream': False}
    with request('/v1/chat/completions', body) as response:
        result = json.load(response)
    answer = result['choices'][0]['message']['content']
    assert answer.strip() and '42' in answer
    assert result['usage']['total_tokens'] > 0
    report['chat'] = {'passed': True, 'answer': answer}
    body['stream'] = True
    chunks, done = [], False
    with request('/v1/chat/completions', body) as response:
        assert 'text/event-stream' in response.headers.get('Content-Type', '')
        for line in response:
            if line.startswith(b'data: '):
                data = line[6:].strip()
                if data == b'[DONE]':
                    done = True
                    break
                chunk = json.loads(data)
                chunks.append(chunk['choices'][0]['delta'].get('content') or '')
    assert done and '42' in ''.join(chunks)
    report['stream'] = {'passed': True, 'done': done, 'answer': ''.join(chunks)}
    with request('/openapi.json') as response:
        current = json.load(response)
    if args.snapshot:
        expected = json.loads(args.snapshot.read_text(encoding='utf-8'))
        assert current == expected, 'OpenAPI changed: review compatibility before refreshing snapshot'
        report['openapi_snapshot'] = 'matched'
    report['passed'] = True
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
